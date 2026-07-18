#!/usr/bin/env luajit

local ok_torch, torch = pcall(require, "torch")
if not ok_torch then error("torch is required") end
local ok_nn, nn = pcall(require, "nn")
if not ok_nn then error("nn is required") end
require("initenv")
require("NeuralQLearner")
local ffi = require("ffi")

local function getenv(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  return value
end

local OUT_ROOT = getenv("OUT", "audit_outputs/stage5_bellman")
local MODEL_DIR = getenv("STAGE5_MODEL_DIR", "audit_outputs/stage5_learner/model_exchange")
local SPEC = getenv("STAGE5B_BATCH_SPEC", OUT_ROOT .. "/bellman_batches.tsv")
local OUT = getenv("DEEPMIND_BELLMAN_OUT", OUT_ROOT .. "/deepmind_bellman.jsonl")
local GAMMA = tonumber(getenv("GAMMA", "0.99"))
local ACTION_COUNT = tonumber(getenv("ACTION_COUNT", "4"))
local HIST_LEN = 4
local FRAME_H = 84
local FRAME_W = 84

local function shell_quote(path) return "'" .. tostring(path):gsub("'", "'\\''") .. "'" end
local function mkdir_p(path) os.execute("mkdir -p " .. shell_quote(path)) end
local function dirname(path) return tostring(path):match("^(.*)/[^/]*$") or "." end

local function json_escape(value)
  value = tostring(value):gsub("\\", "\\\\"):gsub('"', '\\"')
  value = value:gsub("\n", "\\n"):gsub("\r", "\\r"):gsub("\t", "\\t")
  return value
end

local function is_array(tbl)
  if type(tbl) ~= "table" then return false end
  local count, max_key = 0, 0
  for key, _ in pairs(tbl) do
    if type(key) ~= "number" then return false end
    if key > max_key then max_key = key end
    count = count + 1
  end
  return max_key == count
end

local function sorted_keys(tbl)
  local keys = {}
  for key, _ in pairs(tbl) do keys[#keys + 1] = key end
  table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
  return keys
end

local function encode_json(value)
  local value_type = type(value)
  if value_type == "nil" then return "null"
  elseif value_type == "boolean" then return value and "true" or "false"
  elseif value_type == "number" then return tostring(value)
  elseif value_type == "string" then return '"' .. json_escape(value) .. '"'
  elseif value_type == "table" then
    if is_array(value) then
      local parts = {}
      for i = 1, #value do parts[#parts + 1] = encode_json(value[i]) end
      return "[" .. table.concat(parts, ",") .. "]"
    end
    local parts = {}
    for _, key in ipairs(sorted_keys(value)) do
      if value[key] ~= nil then parts[#parts + 1] = '"' .. json_escape(key) .. '":' .. encode_json(value[key]) end
    end
    return "{" .. table.concat(parts, ",") .. "}"
  end
  return '"' .. json_escape(tostring(value)) .. '"'
end

local float_box = ffi.new("float[1]")
local float_bytes = ffi.cast("uint8_t*", float_box)

local function read_float32_raw(path, shape)
  local handle = assert(io.open(path, "rb"))
  local bytes = handle:read("*a")
  handle:close()
  local expected = 1
  for _, dim in ipairs(shape) do expected = expected * dim end
  assert(#bytes == expected * 4, "bad float32 byte count for " .. path)
  local tensor = torch.FloatTensor(unpack(shape))
  local flat = tensor:view(tensor:nElement())
  local byte_ptr = ffi.cast("const uint8_t*", bytes)
  for i = 1, expected do
    float_bytes[0] = byte_ptr[(i - 1) * 4]
    float_bytes[1] = byte_ptr[(i - 1) * 4 + 1]
    float_bytes[2] = byte_ptr[(i - 1) * 4 + 2]
    float_bytes[3] = byte_ptr[(i - 1) * 4 + 3]
    flat[i] = tonumber(float_box[0])
  end
  return tensor:contiguous()
end

local function read_uint8_tensor(path, shape)
  local handle = assert(io.open(path, "rb"))
  local bytes = handle:read("*a")
  handle:close()
  local expected = 1
  for _, dim in ipairs(shape) do expected = expected * dim end
  assert(#bytes == expected, "bad uint8 byte count for " .. path)
  local tensor = torch.ByteTensor(unpack(shape))
  local flat = tensor:view(tensor:nElement())
  for i = 1, expected do flat[i] = string.byte(bytes, i) end
  return tensor:contiguous()
end

local function read_number_list(path, kind)
  local values = {}
  local handle = assert(io.open(path, "r"))
  for line in handle:lines() do
    local stripped = line:match("^%s*(.-)%s*$")
    if stripped ~= "" then values[#values + 1] = tonumber(stripped) end
  end
  handle:close()
  local tensor
  if kind == "long" then tensor = torch.LongTensor(#values)
  elseif kind == "byte" then tensor = torch.ByteTensor(#values)
  else tensor = torch.FloatTensor(#values) end
  for i, value in ipairs(values) do tensor[i] = value end
  return tensor
end

local function split_tsv(line)
  local values = {}
  for value in (line .. "\t"):gmatch("([^\t]*)\t") do values[#values + 1] = value end
  return values
end

local function read_spec(path)
  local rows = {}
  local handle = assert(io.open(path, "r"))
  local header = nil
  for line in handle:lines() do
    if header == nil then
      header = split_tsv(line)
    else
      local values = split_tsv(line)
      local row = {}
      for i, key in ipairs(header) do row[key] = values[i] end
      row.batch_size = tonumber(row.batch_size)
      rows[#rows + 1] = row
    end
  end
  handle:close()
  return rows
end

local function load_model()
  local create_network = require("convnet_atari3")
  local net = create_network({
    hist_len = HIST_LEN,
    ncols = 1,
    input_dims = {HIST_LEN, FRAME_H, FRAME_W},
    n_actions = ACTION_COUNT,
    gpu = -1,
    verbose = 0,
  })
  net:float()
  local defs = {
    {name = "conv1", module_index = 2, weight_shape = {32, 4, 8, 8}, bias_shape = {32}},
    {name = "conv2", module_index = 4, weight_shape = {64, 32, 4, 4}, bias_shape = {64}},
    {name = "conv3", module_index = 6, weight_shape = {64, 64, 3, 3}, bias_shape = {64}},
    {name = "fc1", module_index = 9, weight_shape = {512, 3136}, bias_shape = {512}},
    {name = "fc2", module_index = 11, weight_shape = {ACTION_COUNT, 512}, bias_shape = {ACTION_COUNT}},
  }
  for _, def in ipairs(defs) do
    local module = net.modules[def.module_index]
    module.weight:copy(read_float32_raw(MODEL_DIR .. "/" .. def.name .. "_weight.float32", def.weight_shape))
    module.bias:copy(read_float32_raw(MODEL_DIR .. "/" .. def.name .. "_bias.float32", def.bias_shape))
  end
  return net
end

local function tensor_list(tensor)
  local flat = tensor:float():contiguous():view(tensor:nElement())
  local out = {}
  for i = 1, flat:nElement() do out[i] = tonumber(flat[i]) end
  return out
end

local function row_values(matrix, i)
  local out = {}
  for j = 1, matrix:size(2) do out[j] = tonumber(matrix[i][j]) end
  return out
end

local function tensor_bool_list(tensor)
  local out = {}
  for i = 1, tensor:nElement() do out[i] = tonumber(tensor[i]) ~= 0 end
  return out
end

local function int_tensor_list(tensor, zero_based)
  local out = {}
  for i = 1, tensor:nElement() do
    local value = tonumber(tensor[i])
    if zero_based then value = value - 1 end
    out[i] = value
  end
  return out
end

local function argmax_zero_based(row)
  local best_index = 1
  local best_value = tonumber(row[1])
  for j = 2, row:nElement() do
    local value = tonumber(row[j])
    if value > best_value then
      best_value = value
      best_index = j
    end
  end
  return best_index - 1
end

local function nearly_equal(left, right, tol)
  return math.abs(tonumber(left) - tonumber(right)) <= tol
end

local function rows_for_batch(net, target_net, entry, start_step)
  local batch_size = tonumber(entry.batch_size)
  local states_uint8 = read_uint8_tensor(entry.states_file, {batch_size, HIST_LEN, FRAME_H, FRAME_W})
  local next_states_uint8 = read_uint8_tensor(entry.next_states_file, {batch_size, HIST_LEN, FRAME_H, FRAME_W})
  local s = states_uint8:float():div(255)
  local s2 = next_states_uint8:float():div(255)
  local replay_indices = read_number_list(entry.replay_indices_file, "long")
  local actions_zero = read_number_list(entry.actions_zero_file, "long")
  local actions_one = read_number_list(entry.actions_one_file, "long")
  local rewards = read_number_list(entry.rewards_file, "float")
  local terminals = read_number_list(entry.terminals_file, "byte")
  local true_terminals = read_number_list(entry.true_terminals_file, "byte")
  local life_terminals = read_number_list(entry.life_loss_terminals_file, "byte")

  local learner = {
    network = net,
    target_network = target_net,
    target_q = 1,
    discount = GAMMA,
    clip_delta = nil,
    rescale_r = false,
    r_max = 1,
    gpu = -1,
    minibatch_size = batch_size,
    n_actions = ACTION_COUNT,
  }

  local _, delta, q2_max = dqn.NeuralQLearner.getQUpdate(learner, {
    s = s,
    a = actions_one,
    r = rewards,
    s2 = s2,
    term = terminals,
  })

  local q_values = net:forward(s):float()
  local next_q_values = target_net:forward(s2):float()
  local q_selected = torch.FloatTensor(batch_size)
  for i = 1, batch_size do q_selected[i] = q_values[i][actions_one[i]] end
  local max_next_q = q2_max:float():view(batch_size)
  local continuation_mask = terminals:clone():float():mul(-1):add(1)
  local discounted = max_next_q:clone():mul(GAMMA):cmul(continuation_mask)
  local targets = q_selected:clone():add(delta:float())
  local td_errors = delta:float():clone()

  local rows = {}
  for i = 1, batch_size do
    local terminal_flag = tonumber(terminals[i]) ~= 0
    local target_equals_reward = nearly_equal(targets[i], rewards[i], 1e-7)
    local gamma_applied = (not terminal_flag) and nearly_equal(discounted[i], GAMMA * tonumber(max_next_q[i]), 1e-7)
    rows[#rows + 1] = {
      phase = "bellman_sample",
      source = "deepmind",
      step = start_step + i - 1,
      batch_name = entry.batch_name,
      batch_position = i - 1,
      replay_index = tonumber(replay_indices[i]),
      action = tonumber(actions_zero[i]),
      action_one_based = tonumber(actions_one[i]),
      reward = tonumber(rewards[i]),
      terminal_flag = terminal_flag,
      true_terminal = tonumber(true_terminals[i]) ~= 0,
      life_loss_terminal = tonumber(life_terminals[i]) ~= 0,
      continuation_mask = tonumber(continuation_mask[i]),
      gamma = GAMMA,
      online_q_values = row_values(q_values, i),
      selected_q = tonumber(q_selected[i]),
      target_next_q_values = row_values(next_q_values, i),
      maximizing_next_action = argmax_zero_based(next_q_values[i]),
      max_next_q = tonumber(max_next_q[i]),
      discounted_continuation = tonumber(discounted[i]),
      bellman_target = tonumber(targets[i]),
      td_error = tonumber(td_errors[i]),
      target_network_used = true,
      max_dimension = "action",
      terminal_target_equals_reward = terminal_flag and target_equals_reward,
      nonterminal_gamma_applied = gamma_applied,
    }
  end

  rows[#rows + 1] = {
    phase = "bellman_batch",
    source = "deepmind",
    step = "batch:" .. entry.batch_name,
    batch_name = entry.batch_name,
    batch_size = batch_size,
    replay_indices = tensor_list(replay_indices),
    actions = tensor_list(actions_zero),
    rewards = tensor_list(rewards),
    terminal_flags = tensor_bool_list(terminals),
    maximizing_next_actions = (function()
      local out = {}
      for i = 1, batch_size do out[i] = argmax_zero_based(next_q_values[i]) end
      return out
    end)(),
    targets = tensor_list(targets),
    td_errors = tensor_list(td_errors),
    target_shape = {batch_size},
  }
  return rows
end

mkdir_p(dirname(OUT))

local spec_rows = read_spec(SPEC)
local net = load_model()
local target_net = load_model()

local rows = {}
local step = 0
for _, entry in ipairs(spec_rows) do
  local batch_rows = rows_for_batch(net, target_net, entry, step)
  for _, row in ipairs(batch_rows) do rows[#rows + 1] = row end
  step = step + #batch_rows
end

local out = assert(io.open(OUT, "w"))
for _, row in ipairs(rows) do
  out:write(encode_json(row))
  out:write("\n")
end
out:close()
print("wrote " .. OUT)
print("batches: " .. tostring(#spec_rows))
print("rows: " .. tostring(#rows))
