#!/usr/bin/env luajit

local ok_torch, torch = pcall(require, "torch")
if not ok_torch then error("torch is required") end
local image = require("image")

local function getenv(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  return value
end

local OUT = getenv("RESIZE_DEEPMIND_OUT", getenv("OUT", "audit_outputs/stage2c_resize") .. "/deepmind_outputs/deepmind_resize.jsonl")
local OUTPUT_DIR = getenv("RESIZE_DEEPMIND_OUTPUT_DIR", getenv("OUT", "audit_outputs/stage2c_resize") .. "/deepmind_outputs/arrays")
local FIXTURE_PATHS = getenv("RESIZE_FIXTURE_PATHS", getenv("OUT", "audit_outputs/stage2c_resize") .. "/fixtures/fixture_paths.txt")

local function shell_quote(path)
  return "'" .. tostring(path):gsub("'", "'\\''") .. "'"
end

local function mkdir_p(path)
  os.execute("mkdir -p " .. shell_quote(path))
end

local function dirname(path)
  return tostring(path):match("^(.*)/[^/]*$") or "."
end

local function json_escape(value)
  value = tostring(value)
  value = value:gsub("\\", "\\\\")
  value = value:gsub('"', '\\"')
  value = value:gsub("\n", "\\n")
  value = value:gsub("\r", "\\r")
  value = value:gsub("\t", "\\t")
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
  for key, _ in pairs(tbl) do table.insert(keys, key) end
  table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
  return keys
end

local function encode_json(value)
  local value_type = type(value)
  if value_type == "nil" then
    return "null"
  elseif value_type == "boolean" then
    return value and "true" or "false"
  elseif value_type == "number" then
    return tostring(value)
  elseif value_type == "string" then
    return '"' .. json_escape(value) .. '"'
  elseif value_type == "table" then
    if is_array(value) then
      local parts = {}
      for i = 1, #value do parts[#parts + 1] = encode_json(value[i]) end
      return "[" .. table.concat(parts, ",") .. "]"
    end
    local parts = {}
    for _, key in ipairs(sorted_keys(value)) do
      if value[key] ~= nil then
        parts[#parts + 1] = '"' .. json_escape(key) .. '":' .. encode_json(value[key])
      end
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

local function tensor_bytes(tensor)
  local normalized = tensor:contiguous()
  local flat = normalized:view(normalized:nElement())
  local chunks = {}
  for i = 1, flat:nElement() do
    local value = math.floor(tonumber(flat[i]) + 0.5) % 256
    chunks[#chunks + 1] = string.char(value)
  end
  return table.concat(chunks)
end

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

local function byte_stats(tensor)
  local byte = tensor:contiguous()
  if torch.type(byte) ~= "torch.ByteTensor" then
    byte = byte:clone():clamp(0, 255):byte():contiguous()
  end
  local numeric = byte:double()
  local mean = tonumber(numeric:mean())
  return {
    hash = sha256_bytes(tensor_bytes(byte)),
    shape = tensor_shape(byte),
    dtype = "torch.ByteTensor",
    min = tonumber(numeric:min()),
    max = tonumber(numeric:max()),
    mean = mean,
    std = math.sqrt(tonumber(numeric:clone():add(-mean):pow(2):mean())),
  }
end

local function numeric_stats(tensor)
  local numeric = tensor:double()
  local mean = tonumber(numeric:mean())
  return {
    shape = tensor_shape(tensor),
    type = torch.type(tensor),
    min = tonumber(numeric:min()),
    max = tonumber(numeric:max()),
    mean = mean,
    std = math.sqrt(tonumber(numeric:clone():add(-mean):pow(2):mean())),
  }
end

local function write_npy_uint8(path, tensor)
  local frame = tensor:contiguous()
  if torch.type(frame) ~= "torch.ByteTensor" then frame = frame:byte():contiguous() end
  mkdir_p(dirname(path))
  local shape_parts = {}
  for dim = 1, frame:dim() do shape_parts[#shape_parts + 1] = tostring(frame:size(dim)) end
  if frame:dim() == 1 then shape_parts[1] = shape_parts[1] .. "," end
  local header = "{'descr': '|u1', 'fortran_order': False, 'shape': (" .. table.concat(shape_parts, ", ") .. "), }"
  local magic_len = 10
  local padding = 16 - ((magic_len + #header + 1) % 16)
  if padding == 16 then padding = 0 end
  header = header .. string.rep(" ", padding) .. "\n"
  local handle = assert(io.open(path, "wb"))
  handle:write(string.char(0x93) .. "NUMPY")
  handle:write(string.char(1, 0))
  handle:write(string.char((#header) % 256, math.floor((#header) / 256)))
  handle:write(header)
  handle:write(tensor_bytes(frame))
  handle:close()
end

local function read_npy_uint8(path)
  local handle = assert(io.open(path, "rb"))
  local bytes = handle:read("*a")
  handle:close()
  assert(bytes:sub(1, 6) == string.char(0x93) .. "NUMPY", "not a npy file: " .. path)
  local major = string.byte(bytes, 7)
  assert(major == 1, "only npy v1 is supported")
  local header_len = string.byte(bytes, 9) + 256 * string.byte(bytes, 10)
  local header = bytes:sub(11, 10 + header_len)
  assert(header:find("'descr': '|u1'", 1, true) or header:find([["descr": "|u1"]], 1, true), "only uint8 npy is supported")
  assert(header:find("False", 1, true) or header:find("false", 1, true), "fortran-order npy is unsupported")
  local shape_text = header:match("%(([^%)]*)%)")
  assert(shape_text ~= nil, "missing npy shape")
  local dims = {}
  for dim in shape_text:gmatch("%d+") do dims[#dims + 1] = tonumber(dim) end
  assert(#dims == 2, "expected 2D grayscale fixture")
  local offset = 10 + header_len + 1
  local data = bytes:sub(offset)
  assert(#data == dims[1] * dims[2], "npy payload size mismatch")
  local tensor = torch.ByteTensor(dims[1], dims[2])
  local flat = tensor:view(tensor:nElement())
  for i = 1, #data do flat[i] = string.byte(data, i) end
  return tensor:contiguous()
end

local function read_fixture_paths(path)
  local rows = {}
  local handle = assert(io.open(path, "r"))
  for line in handle:lines() do
    local name, group, frame_path = line:match("^([^\t]+)\t([^\t]+)\t(.+)$")
    if name ~= nil and frame_path ~= nil then
      rows[#rows + 1] = {name = name, group = group, path = frame_path}
    end
  end
  handle:close()
  return rows
end

mkdir_p(dirname(OUT))
mkdir_p(OUTPUT_DIR)
local out = assert(io.open(OUT, "w"))

for _, fixture in ipairs(read_fixture_paths(FIXTURE_PATHS)) do
  local input_byte = read_npy_uint8(fixture.path)
  local input_float = input_byte:float():div(255):view(1, input_byte:size(1), input_byte:size(2)):contiguous()
  local resized = image.scale(input_float, 84, 84, "bilinear"):float()
  if resized:dim() == 3 and resized:size(1) == 1 then resized = resized[1]:contiguous() end
  local output_byte = resized:clone():mul(255):clamp(0, 255):byte():contiguous()
  local output_path = string.format("%s/%s.npy", OUTPUT_DIR, fixture.name)
  write_npy_uint8(output_path, output_byte)

  out:write(encode_json({
    phase = "resize_fixture",
    source = "deepmind",
    fixture_name = fixture.name,
    fixture_group = fixture.group,
    input_path = fixture.path,
    input_hash = byte_stats(input_byte).hash,
    input_tensor = numeric_stats(input_float),
    output_path = output_path,
    output_tensor = numeric_stats(resized),
    output_frame = byte_stats(output_byte),
    resize_source = "torch.image.scale(input_float_1xHxW, 84, 84, 'bilinear')",
    final_cast = "mul(255):clamp(0,255):byte()",
  }))
  out:write("\n")
end

out:close()
print("wrote " .. OUT)
