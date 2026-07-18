#!/usr/bin/env luajit

local ok_torch, torch = pcall(require, "torch")
if not ok_torch then error("torch is required") end
require("initenv")
require("NeuralQLearner")
local ffi = require("ffi")

local function getenv(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  return value
end

local OUT_ROOT = getenv("OUT", "audit_outputs/stage5_loss")
local FIXTURE_TSV = getenv("STAGE5C_FIXTURE_TSV", OUT_ROOT .. "/loss_fixture.tsv")
local OUT = getenv("DEEPMIND_LOSS_OUT", OUT_ROOT .. "/deepmind_loss.jsonl")
local ACTION_COUNT = tonumber(getenv("ACTION_COUNT", "4"))

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

local function sha256_bytes(bytes)
  local tmp = os.tmpname()
  local handle = assert(io.open(tmp, "wb"))
  handle:write(bytes)
  handle:close()
  local proc = assert(io.popen("sha256sum " .. shell_quote(tmp)))
  local line = proc:read("*l")
  proc:close()
  os.remove(tmp)
  return line:match("^(%w+)")
end

local function float32_tensor_hash(tensor)
  local flat = tensor:float():contiguous():view(tensor:nElement())
  local chunks = {}
  for i = 1, flat:nElement() do
    float_box[0] = tonumber(flat[i])
    chunks[#chunks + 1] = string.char(float_bytes[0], float_bytes[1], float_bytes[2], float_bytes[3])
  end
  return sha256_bytes(table.concat(chunks))
end

local function split_tsv(line)
  local values = {}
  for value in (line .. "\t"):gmatch("([^\t]*)\t") do values[#values + 1] = value end
  return values
end

local function read_batches(path)
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
      rows[#rows + 1] = row
    end
  end
  handle:close()

  local batches, by_name, order = {}, {}, {}
  for _, row in ipairs(rows) do
    if by_name[row.batch_name] == nil then
      by_name[row.batch_name] = {}
      order[#order + 1] = row.batch_name
    end
    by_name[row.batch_name][#by_name[row.batch_name] + 1] = row
  end
  for _, name in ipairs(order) do
    table.sort(by_name[name], function(a, b) return tonumber(a.batch_position) < tonumber(b.batch_position) end)
    batches[#batches + 1] = {
      batch_name = name,
      batch_kind = by_name[name][1].batch_kind,
      batch_size = tonumber(by_name[name][1].batch_size),
      samples = by_name[name],
    }
  end
  return batches
end

local StubNetwork = {}
function StubNetwork:new(output)
  local obj = {output = output}
  setmetatable(obj, self)
  self.__index = self
  return obj
end
function StubNetwork:forward(_)
  return self.output
end

local function tensor_list(tensor)
  local flat = tensor:float():contiguous():view(tensor:nElement())
  local out = {}
  for i = 1, flat:nElement() do out[i] = tonumber(flat[i]) end
  return out
end

local function tensor_first_values(tensor, n)
  local flat = tensor:float():contiguous():view(tensor:nElement())
  local limit = math.min(n or 16, flat:nElement())
  local out = {}
  for i = 1, limit do out[i] = tonumber(flat[i]) end
  return out
end

local function row_values(matrix, row)
  local out = {}
  for j = 1, matrix:size(2) do out[j] = tonumber(matrix[row][j]) end
  return out
end

local function tensor_shape(tensor)
  local shape = {}
  for dim = 1, tensor:dim() do shape[#shape + 1] = tensor:size(dim) end
  return shape
end

local function tensor_summary(tensor)
  local data = tensor:float():contiguous()
  return {
    shape = tensor_shape(data),
    dtype = "float32",
    hash = float32_tensor_hash(data),
    first_values = tensor_first_values(data, 16),
    abs_sum = tonumber(data:abs():sum()),
    nonzero_count = tonumber(data:ne(0):sum()),
  }
end

local function abs_sum_tensor(tensor)
  return tonumber(tensor:float():abs():sum())
end

local function huber_losses(raw_td)
  local losses = torch.FloatTensor(raw_td:nElement())
  for i = 1, raw_td:nElement() do
    local value = tonumber(raw_td[i])
    local abs_value = math.abs(value)
    if abs_value <= 1.0 then losses[i] = 0.5 * value * value
    else losses[i] = abs_value - 0.5 end
  end
  return losses
end

local function rows_for_batch(batch, step_start)
  local b = batch.batch_size
  local q_values = torch.FloatTensor(b, ACTION_COUNT):zero()
  local target_q_values = torch.FloatTensor(b, ACTION_COUNT):zero()
  local s = torch.FloatTensor(b, 1):zero()
  local s2 = torch.FloatTensor(b, 1):zero()
  local actions_one = torch.LongTensor(b)
  local actions_zero = torch.LongTensor(b)
  local selected_q = torch.FloatTensor(b)
  local targets = torch.FloatTensor(b)
  local terminals = torch.ByteTensor(b)

  for i, sample in ipairs(batch.samples) do
    local action_zero = tonumber(sample.action)
    local action_one = tonumber(sample.action_one_based)
    actions_zero[i] = action_zero
    actions_one[i] = action_one
    selected_q[i] = tonumber(sample.selected_q)
    targets[i] = tonumber(sample.target)
    terminals[i] = sample.terminal_flag == "1" and 1 or 0
    q_values[i][action_one] = selected_q[i]
  end

  local learner = {
    network = StubNetwork:new(q_values),
    target_network = StubNetwork:new(target_q_values),
    target_q = 1,
    discount = 0,
    clip_delta = 1,
    rescale_r = false,
    r_max = 1,
    gpu = -1,
    minibatch_size = b,
    n_actions = ACTION_COUNT,
  }

  local output_gradient, clipped_delta, _ = dqn.NeuralQLearner.getQUpdate(learner, {
    s = s,
    a = actions_one,
    r = targets,
    s2 = s2,
    term = terminals,
  })

  local raw_td = targets:clone():add(-1, selected_q)
  local pred_minus_target = selected_q:clone():add(-1, targets)
  local abs_td = raw_td:clone():abs()
  local per_sample_loss = huber_losses(raw_td)
  local smooth_l1_mean_dloss = clipped_delta:clone():mul(-1 / b)

  local rows = {}
  for i, sample in ipairs(batch.samples) do
    local row_grad = output_gradient[i]:float()
    local action_zero = tonumber(sample.action)
    local unselected_zero = true
    for j = 1, ACTION_COUNT do
      if j ~= action_zero + 1 and tonumber(row_grad[j]) ~= 0 then unselected_zero = false end
    end
    rows[#rows + 1] = {
      phase = "loss_sample",
      source = "deepmind",
      step = step_start + i - 1,
      batch_name = batch.batch_name,
      batch_kind = batch.batch_kind,
      batch_size = b,
      batch_position = i - 1,
      sample_id = sample.sample_id,
      sample_kind = sample.sample_kind,
      replay_index = tonumber(sample.replay_index),
      action = action_zero,
      action_one_based = tonumber(sample.action_one_based),
      selected_q = tonumber(selected_q[i]),
      target = tonumber(targets[i]),
      raw_td_error = tonumber(raw_td[i]),
      pred_minus_target = tonumber(pred_minus_target[i]),
      abs_td_error = tonumber(abs_td[i]),
      clip_delta = 1.0,
      clipped_td_error = tonumber(clipped_delta[i]),
      strictly_inside_clip_region = tonumber(abs_td[i]) < 1.0,
      at_clip_threshold = tonumber(abs_td[i]) == 1.0,
      per_sample_huber_loss_proxy = tonumber(per_sample_loss[i]),
      selected_action_output_gradient = tonumber(clipped_delta[i]),
      smooth_l1_mean_dloss_d_selected_q = tonumber(smooth_l1_mean_dloss[i]),
      output_gradient_row = row_values(output_gradient, i),
      output_gradient_nonzero_count = tonumber(row_grad:ne(0):sum()),
      unselected_actions_zero = unselected_zero,
      terminal_flag = sample.terminal_flag == "1",
      true_terminal = sample.true_terminal == "1",
      life_loss_terminal = sample.life_loss_terminal == "1",
      deepmind_source_path = "NeuralQLearner:getQUpdate",
      scalar_loss_reported_by_deepmind = "none",
    }
  end

  local nonzero_clipped = tonumber(clipped_delta:ne(0):sum())
  rows[#rows + 1] = {
    phase = "loss_batch",
    source = "deepmind",
    step = "batch:" .. batch.batch_name,
    batch_name = batch.batch_name,
    batch_kind = batch.batch_kind,
    batch_size = b,
    sample_ids = (function()
      local out = {}
      for _, sample in ipairs(batch.samples) do out[#out + 1] = sample.sample_id end
      return out
    end)(),
    actions = tensor_list(actions_zero),
    selected_q_values = tensor_list(selected_q),
    targets = tensor_list(targets),
    raw_td_errors = tensor_list(raw_td),
    clipped_td_errors = tensor_list(clipped_delta),
    per_sample_huber_losses_proxy = tensor_list(per_sample_loss),
    huber_loss_sum_proxy = tonumber(per_sample_loss:sum()),
    huber_loss_mean_proxy = tonumber(per_sample_loss:mean()),
    deepmind_batch_normalization_factor = 1.0,
    deepmind_effective_reduction = "sparse_clipped_delta_no_batch_mean",
    scalar_loss_reported_by_deepmind = "none",
    smooth_l1_mean_scalar_proxy = tonumber(per_sample_loss:mean()),
    smooth_l1_mean_dloss_d_selected_q = tensor_list(smooth_l1_mean_dloss),
    output_gradient_tensor = tensor_summary(output_gradient),
    selected_action_output_gradients = tensor_list(clipped_delta),
    unselected_actions_zero = tonumber(output_gradient:ne(0):sum()) == nonzero_clipped,
  }
  return rows
end

mkdir_p(dirname(OUT))
local rows = {}
local step = 0
for _, batch in ipairs(read_batches(FIXTURE_TSV)) do
  local batch_rows = rows_for_batch(batch, step)
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
print("rows: " .. tostring(#rows))
