#!/usr/bin/env luajit

local ok_torch, torch = pcall(require, "torch")
if not ok_torch then error("torch is required") end
local ok_nn, nn = pcall(require, "nn")
if not ok_nn then error("nn is required") end
require("initenv")
local ffi = require("ffi")

local function getenv(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  return value
end

local OUT_ROOT = getenv("OUT", "audit_outputs/stage5_learner")
local FIXTURE_DIR = getenv("STAGE5_FIXTURE_DIR", OUT_ROOT .. "/learner_fixture")
local MODEL_DIR = getenv("STAGE5_MODEL_DIR", OUT_ROOT .. "/model_exchange")
local OUT = getenv("DEEPMIND_LEARNER_OUT", OUT_ROOT .. "/deepmind_learner.jsonl")
local BATCH_SIZE = tonumber(getenv("STAGE5_BATCH_SIZE", "32"))
local LR = tonumber(getenv("LEARNER_LR", "0.00025"))
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

local function tensor_shape(tensor)
  local shape = {}
  for dim = 1, tensor:dim() do shape[#shape + 1] = tensor:size(dim) end
  return shape
end

local function sha256_file(path)
  local proc = assert(io.popen("sha256sum " .. shell_quote(path)))
  local line = proc:read("*l")
  proc:close()
  return line:match("^(%w+)")
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
  return net, defs
end

local function exchange_layer_contract(defs)
  local rows = {}
  for _, def in ipairs(defs) do
    rows[#rows + 1] = {
      name = def.name,
      weight_shape = def.weight_shape,
      bias_shape = def.bias_shape,
      weight_sha256 = sha256_file(MODEL_DIR .. "/" .. def.name .. "_weight.float32"),
      bias_sha256 = sha256_file(MODEL_DIR .. "/" .. def.name .. "_bias.float32"),
    }
  end
  return rows
end

local function matrix_rows(tensor, rows)
  local out = {}
  local n = math.min(rows or 8, tensor:size(1))
  for i = 1, n do
    out[i] = {}
    for j = 1, tensor:size(2) do out[i][j] = tonumber(tensor[i][j]) end
  end
  return out
end

local function tensor_list(tensor, limit)
  local flat = tensor:float():contiguous():view(tensor:nElement())
  local n = flat:nElement()
  if limit and limit < n then n = limit end
  local out = {}
  for i = 1, n do out[i] = tonumber(flat[i]) end
  return out
end

local function tensor_summary(tensor)
  local data = tensor:float():contiguous()
  return {
    shape = tensor_shape(data),
    dtype = "float32",
    hash = "not_compared",
    mean = tonumber(data:mean()),
    max_abs = tonumber(data:abs():max()),
    l2 = tonumber(torch.sqrt(torch.pow(data:double(), 2):sum())),
    first_values = tensor_list(data, 8),
  }
end

local function layer_summaries(net, include_grads)
  local defs = {
    {name = "conv.0.weight", module_index = 2, field = "weight"},
    {name = "conv.0.bias", module_index = 2, field = "bias"},
    {name = "conv.2.weight", module_index = 4, field = "weight"},
    {name = "conv.2.bias", module_index = 4, field = "bias"},
    {name = "conv.4.weight", module_index = 6, field = "weight"},
    {name = "conv.4.bias", module_index = 6, field = "bias"},
    {name = "fc.0.weight", module_index = 9, field = "weight"},
    {name = "fc.0.bias", module_index = 9, field = "bias"},
    {name = "fc.2.weight", module_index = 11, field = "weight"},
    {name = "fc.2.bias", module_index = 11, field = "bias"},
  }
  local rows = {}
  for _, def in ipairs(defs) do
    local module = net.modules[def.module_index]
    local item = {name = def.name, param = tensor_summary(module[def.field])}
    if include_grads then
      local grad_field = def.field == "weight" and "gradWeight" or "gradBias"
      item.grad = tensor_summary(module[grad_field])
    end
    rows[#rows + 1] = item
  end
  return rows
end

local states_uint8 = read_uint8_tensor(FIXTURE_DIR .. "/states_uint8.bin", {BATCH_SIZE, HIST_LEN, FRAME_H, FRAME_W})
local next_states_uint8 = read_uint8_tensor(FIXTURE_DIR .. "/next_states_uint8.bin", {BATCH_SIZE, HIST_LEN, FRAME_H, FRAME_W})
local s = states_uint8:float():div(255)
local s2 = next_states_uint8:float():div(255)
local a = read_number_list(FIXTURE_DIR .. "/actions_one_based.txt", "long")
local a_zero = read_number_list(FIXTURE_DIR .. "/actions_zero_based.txt", "long")
local r = read_number_list(FIXTURE_DIR .. "/rewards.txt", "float")
local term = read_number_list(FIXTURE_DIR .. "/terminals.txt", "byte")

local net, defs = load_model()
local target_net = net:clone()

local rows = {}
rows[#rows + 1] = {
  phase = "model_contract",
  source = "deepmind",
  step = 0,
  architecture = "convnet_atari3_compatible",
  qnetwork_source_kind = "deepmind_convnet_atari3",
  loss_source_kind = "NeuralQLearner_getQUpdate_and_network_backward",
  optimizer_source_kind = "NeuralQLearner_qLearnMinibatch_manual_centered_rmsprop",
  action_count = ACTION_COUNT,
  input_shape = {4, 84, 84},
  parameter_layers = exchange_layer_contract(defs),
}

local q_values = net:forward(s):float()
local next_q_values = target_net:forward(s2):float()
local max_next_q = next_q_values:max(2):view(BATCH_SIZE)
rows[#rows + 1] = {
  phase = "forward",
  source = "deepmind",
  step = 1,
  input_contract = "effective_float32_0_1_network_input",
  batch_size = BATCH_SIZE,
  q_values_first_rows = matrix_rows(q_values, 8),
  next_q_values_first_rows = matrix_rows(next_q_values, 8),
  q_values_summary = tensor_summary(q_values),
  next_q_values_summary = tensor_summary(next_q_values),
}

local term_mask = term:clone():float():mul(-1):add(1)
local q2 = max_next_q:clone():mul(GAMMA):cmul(term_mask)
local target_values = r:clone():float():add(q2)
local q_selected = torch.FloatTensor(BATCH_SIZE)
for i = 1, BATCH_SIZE do q_selected[i] = q_values[i][a[i]] end
local delta_unclipped = target_values:clone():add(-1, q_selected)
local clipped_delta = delta_unclipped:clone()
clipped_delta[clipped_delta:ge(1)] = 1
clipped_delta[clipped_delta:le(-1)] = -1
local sparse_targets = torch.zeros(BATCH_SIZE, ACTION_COUNT):float()
for i = 1, BATCH_SIZE do sparse_targets[i][a[i]] = clipped_delta[i] end

rows[#rows + 1] = {
  phase = "bellman_target",
  source = "deepmind",
  step = 2,
  gamma = GAMMA,
  actions_zero_based = tensor_list(a_zero),
  actions_one_based = tensor_list(a),
  rewards = tensor_list(r),
  terminals = tensor_list(term),
  q_selected = tensor_list(q_selected),
  max_next_q = tensor_list(max_next_q),
  target_values = tensor_list(target_values),
  delta_unclipped = tensor_list(delta_unclipped),
  clipped_delta = tensor_list(clipped_delta),
}

local w, dw = net:getParameters()
dw:zero()
net:backward(s, sparse_targets)
local grad_norm = tonumber(torch.sqrt(torch.pow(dw:double(), 2):sum()))

rows[#rows + 1] = {
  phase = "loss_gradient_contract",
  source = "deepmind",
  step = 3,
  loss_mode = "deepmind_sparse_clipped_delta_backward_no_scalar_loss",
  scalar_loss = "none",
  output_gradient_scale = "d_network_output_is_sparse_clipped_delta_no_batch_mean",
  grad_norm = grad_norm,
  pred_minus_target_first_values = tensor_list(q_selected:clone():add(-1, target_values), 8),
}
rows[#rows + 1] = {
  phase = "gradient_summary",
  source = "deepmind",
  step = 4,
  parameter_layers = layer_summaries(net, true),
}

local before = w:clone()
local g = dw:clone():fill(0)
local g2 = dw:clone():fill(0)
local tmp = dw:clone():fill(0)
local deltas = dw:clone():fill(0)
g:mul(0.95):add(0.05, dw)
tmp:cmul(dw, dw)
g2:mul(0.95):add(0.05, tmp)
tmp:cmul(g, g)
tmp:mul(-1)
tmp:add(g2)
tmp:add(0.01)
tmp:sqrt()
deltas:mul(0):addcdiv(LR, dw, tmp)
w:add(deltas)
local param_delta = w:clone():add(-1, before)

rows[#rows + 1] = {
  phase = "optimizer_update",
  source = "deepmind",
  step = 5,
  optimizer_mode = "NeuralQLearner_manual_centered_rmsprop_on_sparse_clipped_delta_gradients",
  lr = LR,
  alpha = 0.95,
  epsilon = 0.01,
  epsilon_placement = "inside_sqrt",
  flat_parameter_delta = tensor_summary(param_delta),
  post_update_layers = layer_summaries(net, false),
}

mkdir_p(dirname(OUT))
local out = assert(io.open(OUT, "w"))
for _, row in ipairs(rows) do
  out:write(encode_json(row))
  out:write("\n")
end
out:close()
print("wrote " .. OUT)
print("rows: " .. tostring(#rows))
