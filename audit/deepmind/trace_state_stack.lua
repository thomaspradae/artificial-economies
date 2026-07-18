#!/usr/bin/env luajit

local ok_torch, torch = pcall(require, "torch")
if not ok_torch then error("torch is required") end

local function getenv(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  return value
end

local OUT_ROOT = getenv("OUT", "audit_outputs/stage3_replay")
local TAPE_DIR = getenv("STAGE3_CANONICAL_DIR", OUT_ROOT .. "/canonical")
local OUT = getenv("DEEPMIND_STATE_STACK_OUT", OUT_ROOT .. "/deepmind_state_stack.jsonl")
local HIST_LEN = tonumber(getenv("HIST_LEN", "4"))
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
  for key, _ in pairs(tbl) do keys[#keys + 1] = key end
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
  for i = 1, flat:nElement() do chunks[#chunks + 1] = string.char(tonumber(flat[i]) % 256) end
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

local function tensor_hash(tensor)
  return sha256_bytes(tensor_bytes(tensor:contiguous()))
end

local function arr_stats(tensor)
  local byte = tensor:contiguous()
  local numeric = byte:double()
  local mean = tonumber(numeric:mean())
  return {
    hash = tensor_hash(byte),
    shape = tensor_shape(byte),
    dtype = "uint8",
    min = tonumber(numeric:min()),
    max = tonumber(numeric:max()),
    mean = mean,
    std = math.sqrt(tonumber(numeric:clone():add(-mean):pow(2):mean())),
  }
end

local function read_npy_uint8_2d(path)
  local handle = assert(io.open(path, "rb"))
  local bytes = handle:read("*a")
  handle:close()
  assert(bytes:sub(1, 6) == string.char(0x93) .. "NUMPY", "not npy: " .. path)
  local header_len = string.byte(bytes, 9) + 256 * string.byte(bytes, 10)
  local header = bytes:sub(11, 10 + header_len)
  assert(header:find("'descr': '|u1'", 1, true) or header:find([["descr": "|u1"]], 1, true), "expected uint8 npy")
  local shape_text = header:match("%(([^%)]*)%)")
  local dims = {}
  for dim in shape_text:gmatch("%d+") do dims[#dims + 1] = tonumber(dim) end
  assert(#dims == 2 and dims[1] == 84 and dims[2] == 84, "expected 84x84 npy: " .. path)
  local offset = 10 + header_len + 1
  local data = bytes:sub(offset)
  local tensor = torch.ByteTensor(84, 84)
  local flat = tensor:view(tensor:nElement())
  for i = 1, #data do flat[i] = string.byte(data, i) end
  return tensor:contiguous()
end

local function split_tsv(line)
  local values = {}
  for value in (line .. "\t"):gmatch("([^\t]*)\t") do values[#values + 1] = value end
  return values
end

local function read_transitions(path)
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
      rows[#rows + 1] = {
        transition_index = tonumber(row.transition_index),
        frame_index = tonumber(row.frame_index),
        frame_path = row.frame_path,
        processed_frame_hash = row.processed_frame_hash,
        action_index = tonumber(row.action_index),
        ale_action_code = tonumber(row.ale_action_code),
        raw_reward = tonumber(row.raw_reward),
        clipped_reward = tonumber(row.clipped_reward),
        true_terminal = row.true_terminal == "1",
        life_loss = row.life_loss == "1",
        terminal_mask = row.terminal_mask == "1",
        lives_before = tonumber(row.lives_before),
        lives_after = tonumber(row.lives_after),
      }
    end
  end
  handle:close()
  return rows
end

local function zero_record(zero)
  return {
    frame_index = nil,
    frame = zero,
    hash = tensor_hash(zero),
    true_terminal = true,
    life_loss = false,
    terminal_mask = true,
  }
end

local function stack_from_recent(recent, zero)
  local zero_flags = {}
  for i = 1, #recent do zero_flags[i] = false end
  local zero_out = false
  for i = #recent - 1, 1, -1 do
    if not zero_out and recent[i].terminal_mask then zero_out = true end
    if zero_out then zero_flags[i] = true end
  end

  local stack = torch.ByteTensor(HIST_LEN, 84, 84):zero()
  local component_indices = {}
  local component_source_indices = {}
  local component_hashes = {}
  local component_source_hashes = {}
  local zeroed_positions = {}
  local episode_boundary = false
  local life_loss_boundary = false
  local true_terminal_boundary = false

  for i = 1, HIST_LEN do
    component_source_indices[i] = recent[i].frame_index or JSON_NULL
    component_source_hashes[i] = recent[i].hash
    if i < HIST_LEN then
      if recent[i].terminal_mask then episode_boundary = true end
      if recent[i].life_loss then life_loss_boundary = true end
      if recent[i].true_terminal then true_terminal_boundary = true end
    end
    if zero_flags[i] or recent[i].frame_index == nil then
      stack[i]:copy(zero)
      component_indices[i] = JSON_NULL
      zeroed_positions[#zeroed_positions + 1] = i - 1
    else
      stack[i]:copy(recent[i].frame)
      component_indices[i] = recent[i].frame_index
    end
    component_hashes[i] = tensor_hash(stack[i])
  end

  return {
    stack = stack:contiguous(),
    component_frame_indices = component_indices,
    component_source_frame_indices = component_source_indices,
    component_frame_hashes = component_hashes,
    component_source_frame_hashes = component_source_hashes,
    reset_zero_padding = #zeroed_positions > 0,
    zeroed_component_positions = zeroed_positions,
    episode_boundary = episode_boundary,
    life_loss_boundary = life_loss_boundary,
    true_terminal_boundary = true_terminal_boundary,
  }
end

local transitions = read_transitions(TAPE_DIR .. "/transitions.tsv")
local zero = torch.ByteTensor(84, 84):zero()
local recent = {}
for i = 1, HIST_LEN do recent[#recent + 1] = zero_record(zero) end
local rows = {}

for _, transition in ipairs(transitions) do
  local frame = read_npy_uint8_2d(transition.frame_path)
  recent[#recent + 1] = {
    frame_index = transition.frame_index,
    frame = frame,
    hash = tensor_hash(frame),
    true_terminal = transition.true_terminal,
    life_loss = transition.life_loss,
    terminal_mask = transition.terminal_mask,
  }
  while #recent > HIST_LEN do table.remove(recent, 1) end
  local info = stack_from_recent(recent, zero)
  local stack = info.stack
  info.stack = nil
  rows[#rows + 1] = {
    phase = "state_stack",
    source = "deepmind",
    step = transition.transition_index,
    transition_index = transition.transition_index,
    frame_index = transition.frame_index,
    current_frame_hash = transition.processed_frame_hash,
    stack_order = "oldest_to_newest",
    hist_len = HIST_LEN,
    state_shape = tensor_shape(stack),
    state_dtype = "uint8",
    state_hash = tensor_hash(stack),
    state_frame = arr_stats(stack),
    component_frame_indices = info.component_frame_indices,
    component_source_frame_indices = info.component_source_frame_indices,
    component_frame_hashes = info.component_frame_hashes,
    component_source_frame_hashes = info.component_source_frame_hashes,
    reset_zero_padding = info.reset_zero_padding,
    zeroed_component_positions = info.zeroed_component_positions,
    episode_boundary = info.episode_boundary,
    life_loss_boundary = info.life_loss_boundary,
    true_terminal_boundary = info.true_terminal_boundary,
  }
end

for i = 1, #rows do
  if i < #rows then
    rows[i].next_state_component_hashes = rows[i + 1].component_frame_hashes
    rows[i].next_state_hash = rows[i + 1].state_hash
    rows[i].next_state_shape = rows[i + 1].state_shape
    rows[i].next_state_dtype = rows[i + 1].state_dtype
  else
    rows[i].next_state_component_hashes = JSON_NULL
    rows[i].next_state_hash = JSON_NULL
    rows[i].next_state_shape = JSON_NULL
    rows[i].next_state_dtype = JSON_NULL
  end
end

mkdir_p(dirname(OUT))
local out = assert(io.open(OUT, "w"))
for _, row in ipairs(rows) do out:write(encode_json(row)); out:write("\n") end
out:close()
print("wrote " .. OUT)
print("state_stack_rows: " .. tostring(#rows))
