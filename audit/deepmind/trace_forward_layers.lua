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
local OUT = getenv("DEEPMIND_FORWARD_LAYERS_OUT", OUT_ROOT .. "/deepmind_forward_layers.jsonl")
local TENSOR_DIR = getenv("DEEPMIND_FORWARD_TENSOR_DIR", OUT_ROOT .. "/forward_layers/deepmind")
local ACTION_COUNT = tonumber(getenv("ACTION_COUNT", "4"))
local BATCH_SIZE = tonumber(getenv("STAGE5_BATCH_SIZE", "32"))
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

local float_box = ffi.new("float[1]")
local float_bytes = ffi.cast("uint8_t*", float_box)

local function write_float32_raw(path, tensor)
  local flat = tensor:float():contiguous():view(tensor:nElement())
  local out = assert(io.open(path, "wb"))
  for i = 1, flat:nElement() do
    float_box[0] = tonumber(flat[i])
    out:write(string.char(float_bytes[0], float_bytes[1], float_bytes[2], float_bytes[3]))
  end
  out:close()
end

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

local function tensor_stats(tensor, rel_path)
  local data = tensor:float():contiguous()
  local flat = data:view(data:nElement())
  local first = {}
  for i = 1, math.min(16, flat:nElement()) do first[i] = tonumber(flat[i]) end
  return {
    tensor_file = rel_path,
    shape = tensor_shape(data),
    dtype = "float32",
    hash = "not_compared",
    min = tonumber(data:min()),
    max = tonumber(data:max()),
    mean = tonumber(data:mean()),
    first_values = first,
  }
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

local layer_modules = {
  {name = "conv1_pre", module_index = 2},
  {name = "conv1_post", module_index = 3},
  {name = "conv2_pre", module_index = 4},
  {name = "conv2_post", module_index = 5},
  {name = "conv3_pre", module_index = 6},
  {name = "conv3_post", module_index = 7},
  {name = "flatten", module_index = 8},
  {name = "fc1_pre", module_index = 9},
  {name = "fc1_post", module_index = 10},
  {name = "q_values", module_index = 11},
}

local function trace_prefix(rows, net, source_name, input, step_start)
  net:forward(input)
  local rel_input = source_name .. "_input.float32"
  write_float32_raw(TENSOR_DIR .. "/" .. rel_input, input)
  local row = {
    phase = "forward_layer",
    source = "deepmind",
    step = step_start,
    batch_source = source_name,
    layer = "input",
  }
  for k, v in pairs(tensor_stats(input, rel_input)) do row[k] = v end
  rows[#rows + 1] = row
  for offset, item in ipairs(layer_modules) do
    local tensor = net.modules[item.module_index].output:float():contiguous()
    local rel = source_name .. "_" .. item.name .. ".float32"
    write_float32_raw(TENSOR_DIR .. "/" .. rel, tensor)
    local layer_row = {
      phase = "forward_layer",
      source = "deepmind",
      step = step_start + offset,
      batch_source = source_name,
      layer = item.name,
    }
    for k, v in pairs(tensor_stats(tensor, rel)) do layer_row[k] = v end
    rows[#rows + 1] = layer_row
  end
end

mkdir_p(TENSOR_DIR)
mkdir_p(dirname(OUT))

local states_uint8 = read_uint8_tensor(FIXTURE_DIR .. "/states_uint8.bin", {BATCH_SIZE, HIST_LEN, FRAME_H, FRAME_W})
local next_states_uint8 = read_uint8_tensor(FIXTURE_DIR .. "/next_states_uint8.bin", {BATCH_SIZE, HIST_LEN, FRAME_H, FRAME_W})
local states = states_uint8:float():div(255)
local next_states = next_states_uint8:float():div(255)
local net = load_model()

local rows = {
  {
    phase = "forward_architecture",
    source = "deepmind",
    step = 0,
    network_class = "convnet_atari3",
    architecture_manifest = "DeepMind Torch7 convnet_atari3",
    conv1_padding = {1, 1},
    weight_mapping = "deepmind_raw_to_torch7_modules",
  },
}
trace_prefix(rows, net, "state", states, 1)
trace_prefix(rows, net, "next_state", next_states, 100)

local out = assert(io.open(OUT, "w"))
for _, row in ipairs(rows) do
  out:write(encode_json(row))
  out:write("\n")
end
out:close()
print("wrote " .. OUT)
print("rows: " .. tostring(#rows))
