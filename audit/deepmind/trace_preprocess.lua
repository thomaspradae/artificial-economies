#!/usr/bin/env luajit

-- Apply DeepMind's net_downsample_2x_full_y to a frozen frame tape.

local ok_torch, torch = pcall(require, "torch")
if not ok_torch then error("torch is required") end
local image = require("image")

local function getenv(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  return value
end

local TAPE_DIR = getenv("CANONICAL_TAPE_DIR", getenv("OUT", "audit_outputs") .. "/canonical_frames")
local OUT = getenv("DEEPMIND_PREPROCESS_OUT", getenv("OUT", "audit_outputs") .. "/deepmind_preprocess.jsonl")
local PROCESSED_DIR = getenv("DEEPMIND_PROCESSED_DIR", getenv("OUT", "audit_outputs") .. "/deepmind_processed")
local POOLED_T7_PATHS = getenv("POOLED_T7_PATHS", TAPE_DIR .. "/pooled_t7_paths.txt")
local DM_DIR = getenv("DM_DIR", ".")
local SOURCE_REPORT_OUT = getenv("DEEPMIND_PREPROCESS_SOURCE_OUT", getenv("OUT", "audit_outputs") .. "/deepmind_preprocess_source.txt")
local INTERMEDIATE_DIR = getenv("DEEPMIND_PREPROCESS_INTERMEDIATE_DIR", getenv("OUT", "audit_outputs") .. "/deepmind_preprocess_intermediates")
local INTERMEDIATE_DUMP_STEPS = tonumber(getenv("DEEPMIND_PREPROCESS_DUMP_STEPS", "3"))

local JSON_NULL = {__json_null = true}

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
    if value == JSON_NULL then return "null" end
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

local function file_sha256(path)
  local handle = io.open(path, "rb")
  if handle == nil then return nil end
  local bytes = handle:read("*a")
  handle:close()
  return sha256_bytes(bytes)
end

local function matching_lines(path, patterns)
  local rows = {}
  local handle = io.open(path, "r")
  if handle == nil then return rows end
  local line_number = 0
  for line in handle:lines() do
    line_number = line_number + 1
    for _, pattern in ipairs(patterns) do
      if line:find(pattern, 1, true) then
        rows[#rows + 1] = string.format("%d:%s", line_number, line)
        break
      end
    end
  end
  handle:close()
  return rows
end

local function write_source_report(path)
  mkdir_p(dirname(path))
  local handle = assert(io.open(path, "w"))
  handle:write("DeepMind preprocessing source\n\n")
  handle:write("Network: net_downsample_2x_full_y\n")
  handle:write("Call chain: create_network -> nn.Scale(84, 84, true) -> nn.Scale:forward -> image.rgb2y -> image.scale(..., 'bilinear')\n\n")
  handle:write("Note: image.rgb2y dispatches to input.image.rgb2y and image.scale(..., 'bilinear') dispatches to src.image.scaleBilinear. In this installed DeepMind/Torch tree those are compiled native extension entry points; C source files were not present under the installation.\n\n")

  local files = {
    {
      path = DM_DIR .. "/dqn/net_downsample_2x_full_y.lua",
      functions = "create_network",
      patterns = {"create_network", "nn.Scale"},
    },
    {
      path = DM_DIR .. "/dqn/Scale.lua",
      functions = "nn.Scale:forward",
      patterns = {"torch.class('nn.Scale'", "function scale:forward", "image.rgb2y", "image.scale"},
    },
    {
      path = DM_DIR .. "/torch/share/lua/5.1/image/init.lua",
      functions = "image.rgb2y, image.scale, image.scaleBilinear dispatch",
      patterns = {"function image.scale", "scaleBilinear", "function image.rgb2y", "input.image.rgb2y"},
    },
  }

  for _, file in ipairs(files) do
    handle:write("=== " .. file.path .. " ===\n")
    handle:write("functions: " .. file.functions .. "\n")
    local digest = file_sha256(file.path)
    handle:write("sha256: " .. tostring(digest) .. "\n")
    local lines = matching_lines(file.path, file.patterns)
    if #lines == 0 then
      handle:write("relevant lines: unavailable\n")
    else
      handle:write("relevant lines:\n")
      for _, line in ipairs(lines) do
        handle:write(line .. "\n")
      end
    end
    handle:write("\n")
  end

  handle:close()
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

local function chw_to_hwc_byte(chw)
  local byte = chw:contiguous()
  if torch.type(byte) ~= "torch.ByteTensor" then byte = byte:clamp(0, 255):byte():contiguous() end
  local channels = byte:size(1)
  local height = byte:size(2)
  local width = byte:size(3)
  local out = torch.ByteTensor(height, width, channels)
  for y = 1, height do
    for x = 1, width do
      for c = 1, channels do
        out[y][x][c] = byte[c][y][x]
      end
    end
  end
  return out:contiguous()
end

local function squeeze_single_channel(tensor)
  local out = tensor
  if out:dim() == 3 and out:size(1) == 1 then
    out = out[1]
  end
  return out:contiguous()
end

local function unit_float_to_byte_trunc(tensor)
  return squeeze_single_channel(tensor):clone():mul(255):clamp(0, 255):byte():contiguous()
end

local function dump_intermediates(step, raw_chw, input, processed)
  if step > INTERMEDIATE_DUMP_STEPS then return end
  mkdir_p(INTERMEDIATE_DIR)

  local raw_hwc_byte = chw_to_hwc_byte(raw_chw)
  local luminance = image.rgb2y(input)
  local resized = image.scale(luminance, 84, 84, "bilinear")

  write_npy_uint8(string.format("%s/raw_hwc_%06d.npy", INTERMEDIATE_DIR, step), raw_hwc_byte)
  write_npy_uint8(string.format("%s/luminance_%06d.npy", INTERMEDIATE_DIR, step), unit_float_to_byte_trunc(luminance))
  write_npy_uint8(string.format("%s/resized_%06d.npy", INTERMEDIATE_DIR, step), unit_float_to_byte_trunc(resized))
  write_npy_uint8(string.format("%s/final_network_input_%06d.npy", INTERMEDIATE_DIR, step), unit_float_to_byte_trunc(processed))

  torch.save(string.format("%s/luminance_float_%06d.t7", INTERMEDIATE_DIR, step), squeeze_single_channel(luminance):float())
  torch.save(string.format("%s/resized_float_%06d.t7", INTERMEDIATE_DIR, step), squeeze_single_channel(resized):float())
  torch.save(string.format("%s/final_network_input_float_%06d.t7", INTERMEDIATE_DIR, step), squeeze_single_channel(processed):float())
end

local function read_paths(path)
  local rows = {}
  local handle = assert(io.open(path, "r"))
  for line in handle:lines() do
    local step, frame_path = line:match("^(%d+)%s+(.+)$")
    if step ~= nil and frame_path ~= nil then
      rows[#rows + 1] = {step = tonumber(step), path = frame_path}
    end
  end
  handle:close()
  return rows
end

local create_preproc = require("net_downsample_2x_full_y")
local preproc = create_preproc()
preproc:float()

mkdir_p(dirname(OUT))
mkdir_p(PROCESSED_DIR)
write_source_report(SOURCE_REPORT_OUT)
local out = assert(io.open(OUT, "w"))

for _, row in ipairs(read_paths(POOLED_T7_PATHS)) do
  local raw_chw = torch.load(row.path)
  local input = raw_chw:float():div(255)
  local processed = preproc:forward(input):float()
  dump_intermediates(row.step, raw_chw, input, processed)
  if processed:dim() == 3 and processed:size(1) == 1 then
    processed = processed[1]:contiguous()
  else
    processed = processed:contiguous()
  end
  -- DeepMind replay insertion multiplies preprocessed float frames by 255 and
  -- stores them in ByteTensors, which truncates fractional values.
  local processed_byte = processed:clone():mul(255):clamp(0, 255):byte():contiguous()
  local processed_path = string.format("%s/processed_%06d.npy", PROCESSED_DIR, row.step)
  write_npy_uint8(processed_path, processed_byte)
  local output = {
    phase = "preprocess",
    source = "deepmind",
    step = row.step,
    input_t7_path = row.path,
    processed_path = processed_path,
    processed_frame = byte_stats(processed_byte),
    preprocess_source = "net_downsample_2x_full_y",
    resize_interpolation = "bilinear",
    source_report_path = SOURCE_REPORT_OUT,
    intermediate_dir = INTERMEDIATE_DIR,
  }
  out:write(encode_json(output))
  out:write("\n")
end

out:close()
print("wrote " .. OUT)
