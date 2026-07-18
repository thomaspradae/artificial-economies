#!/usr/bin/env luajit

local ok_torch, torch = pcall(require, "torch")
if not ok_torch then error("torch is required") end
local ok_nn, nn = pcall(require, "nn")
if not ok_nn then error("nn is required") end
local ffi = require("ffi")

local function getenv(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  return value
end

local OUT_ROOT = getenv("OUT", "audit_outputs/stage5_learner")
local MODEL_DIR = getenv("STAGE5_MODEL_DIR", OUT_ROOT .. "/model_exchange")
local SEED = tonumber(getenv("MODEL_SEED", "1"))
local ACTIONS = tonumber(getenv("ACTION_COUNT", "4"))

local function shell_quote(path) return "'" .. tostring(path):gsub("'", "'\\''") .. "'" end
local function mkdir_p(path) os.execute("mkdir -p " .. shell_quote(path)) end

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

local function write_float32_raw(path, tensor)
  local flat = tensor:float():contiguous():view(tensor:nElement())
  local out = assert(io.open(path, "wb"))
  for i = 1, flat:nElement() do
    float_box[0] = tonumber(flat[i])
    out:write(string.char(float_bytes[0], float_bytes[1], float_bytes[2], float_bytes[3]))
  end
  out:close()
end

mkdir_p(MODEL_DIR)
torch.manualSeed(SEED)

local create_network = require("convnet_atari3")
local net = create_network({
  hist_len = 4,
  ncols = 1,
  input_dims = {4, 84, 84},
  n_actions = ACTIONS,
  gpu = -1,
  verbose = 0,
})
net:float()

local layer_names = {
  "conv1",
  "conv2",
  "conv3",
  "fc1",
  "fc2",
}
local layers = {}
local named_index = 1
for module_index, module in ipairs(net.modules) do
  if module.weight ~= nil and module.bias ~= nil then
    local name = layer_names[named_index] or ("layer" .. tostring(named_index))
    local weight_file = name .. "_weight.float32"
    local bias_file = name .. "_bias.float32"
    write_float32_raw(MODEL_DIR .. "/" .. weight_file, module.weight)
    write_float32_raw(MODEL_DIR .. "/" .. bias_file, module.bias)
    layers[#layers + 1] = {
      name = name,
      module_index = module_index,
      module_type = torch.typename(module),
      weight_file = weight_file,
      weight_shape = tensor_shape(module.weight),
      weight_sha256 = sha256_file(MODEL_DIR .. "/" .. weight_file),
      bias_file = bias_file,
      bias_shape = tensor_shape(module.bias),
      bias_sha256 = sha256_file(MODEL_DIR .. "/" .. bias_file),
    }
    named_index = named_index + 1
  end
end

local manifest = {
  phase = "stage5_deepmind_model_export",
  source = "DeepMind convnet_atari3",
  model_seed = SEED,
  action_count = ACTIONS,
  input_shape = {4, 84, 84},
  layer_count = #layers,
  layers = layers,
  torch_default_tensor_type = torch.typename(torch.Tensor()),
  network = tostring(net),
}

local manifest_path = MODEL_DIR .. "/deepmind_model_manifest.json"
local out = assert(io.open(manifest_path, "w"))
out:write(encode_json(manifest))
out:write("\n")
out:close()

torch.save(MODEL_DIR .. "/deepmind_model.t7", net)
print("wrote " .. manifest_path)
print("wrote " .. MODEL_DIR .. "/deepmind_model.t7")
print("layers: " .. tostring(#layers))
