#!/usr/bin/env luajit

local ok_torch, torch = pcall(require, "torch")
if not ok_torch then error("torch is required") end
local ffi = require("ffi")

local function getenv(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  return value
end

local OUT_ROOT = getenv("OUT", "audit_outputs/stage4_replay_sample")
local REPLAY_DIR = getenv("STAGE4_REPLAY_DIR", OUT_ROOT .. "/canonical_replay")
local REQUESTED = getenv("STAGE4_REQUESTED_INDICES", OUT_ROOT .. "/requested_indices.txt")
local OUT = getenv("DEEPMIND_REPLAY_SAMPLE_OUT", OUT_ROOT .. "/deepmind_batch.jsonl")
local HIST_LEN = 4
local JSON_NULL = {__json_null = true}

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
    if value == JSON_NULL then return "null" end
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

local function byte_tensor_bytes(tensor)
  local flat = tensor:contiguous():view(tensor:nElement())
  local chunks = {}
  for i = 1, flat:nElement() do chunks[#chunks + 1] = string.char(tonumber(flat[i]) % 256) end
  return table.concat(chunks)
end

local float_box = ffi.new("float[1]")
local float_bytes = ffi.cast("uint8_t*", float_box)

local function float32_normalized_bytes(tensor)
  local flat = tensor:contiguous():view(tensor:nElement())
  local chunks = {}
  for i = 1, flat:nElement() do
    float_box[0] = tonumber(flat[i]) / 255.0
    chunks[#chunks + 1] = string.char(float_bytes[0], float_bytes[1], float_bytes[2], float_bytes[3])
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

local function tensor_hash(tensor) return sha256_bytes(byte_tensor_bytes(tensor)) end
local function normalized_hash(tensor) return sha256_bytes(float32_normalized_bytes(tensor)) end

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

local function read_records(path)
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
        replay_index = tonumber(row.replay_index),
        absolute_insert_index = tonumber(row.absolute_insert_index),
        frame_path = row.frame_path,
        processed_frame_hash = row.processed_frame_hash,
        action = tonumber(row.action),
        action_one_based = tonumber(row.action_one_based),
        ale_action_code = tonumber(row.ale_action_code),
        raw_reward = tonumber(row.raw_reward),
        clipped_reward = tonumber(row.clipped_reward),
        true_terminal = row.true_terminal == "1",
        life_loss_terminal = row.life_loss_terminal == "1",
        terminal_mask = row.terminal_mask == "1",
        episode_id = tonumber(row.episode_id),
        frame_position_in_episode = tonumber(row.frame_position_in_episode),
        sampleable_as_start = row.sampleable_as_start == "1",
        sampleability_reason = row.sampleability_reason,
        wrap_insert_index = tonumber(row.wrap_insert_index),
      }
    end
  end
  handle:close()
  table.sort(rows, function(a, b) return a.replay_index < b.replay_index end)
  return rows
end

local function read_indices(path)
  local rows = {}
  local handle = assert(io.open(path, "r"))
  for line in handle:lines() do
    local stripped = line:match("^%s*(.-)%s*$")
    if stripped ~= "" and stripped:sub(1, 1) ~= "#" then rows[#rows + 1] = tonumber(stripped:match("^-?%d+")) end
  end
  handle:close()
  return rows
end

local function validity(records, requested)
  local max_start = #records - HIST_LEN - 1
  if requested < 1 then return false, "before_earliest_deepmind_sample_index" end
  if requested > max_start then return false, "after_latest_deepmind_sample_index" end
  local action_record = records[requested + HIST_LEN]
  if action_record.terminal_mask then return false, "terminal_at_action_frame" end
  return true, "accepted"
end

local function stack_components(records, start)
  local zero = torch.ByteTensor(84, 84):zero()
  local raw_indices, zero_flags = {}, {}
  for i = 1, HIST_LEN do
    raw_indices[i] = start + i - 1
    zero_flags[i] = false
  end
  local zero_out = false
  for pos = HIST_LEN - 1, 1, -1 do
    if not zero_out and records[raw_indices[pos] + 1].terminal_mask then zero_out = true end
    if zero_out then zero_flags[pos] = true end
  end
  local stack = torch.ByteTensor(HIST_LEN, 84, 84):zero()
  local source_indices, source_hashes, raw_hashes, zeroed_positions = {}, {}, {}, {}
  for pos = 1, HIST_LEN do
    local replay_index = raw_indices[pos]
    local record = records[replay_index + 1]
    raw_hashes[pos] = record.processed_frame_hash
    if zero_flags[pos] then
      stack[pos]:copy(zero)
      source_indices[pos] = JSON_NULL
      source_hashes[pos] = tensor_hash(zero)
      zeroed_positions[#zeroed_positions + 1] = pos - 1
    else
      local frame = read_npy_uint8_2d(record.frame_path)
      stack[pos]:copy(frame)
      source_indices[pos] = replay_index
      source_hashes[pos] = tensor_hash(frame)
    end
  end
  return {
    stack = stack:contiguous(),
    raw_frame_indices = raw_indices,
    source_frame_indices = source_indices,
    source_frame_hashes = source_hashes,
    raw_frame_hashes = raw_hashes,
    zeroed_positions = zeroed_positions,
  }
end

local function sample_at(records, requested)
  local accepted, reason = validity(records, requested)
  local base = {
    requested_replay_index = requested,
    accepted = accepted,
    rejection_reason = accepted and "none" or reason,
  }
  if not accepted then return base end
  local state = stack_components(records, requested)
  local next_state = stack_components(records, requested + 1)
  local action_record = records[requested + HIST_LEN]
  local terminal_record = records[requested + HIST_LEN + 1]
  base.actual_replay_index_used = requested
  base.state_raw_frame_indices = state.raw_frame_indices
  base.state_source_frame_indices = state.source_frame_indices
  base.next_state_raw_frame_indices = next_state.raw_frame_indices
  base.next_state_source_frame_indices = next_state.source_frame_indices
  base.state_component_frame_hashes = state.source_frame_hashes
  base.next_state_component_frame_hashes = next_state.source_frame_hashes
  base.state_raw_frame_hashes = state.raw_frame_hashes
  base.next_state_raw_frame_hashes = next_state.raw_frame_hashes
  base.state_zeroed_positions = state.zeroed_positions
  base.next_state_zeroed_positions = next_state.zeroed_positions
  base.state_hash = tensor_hash(state.stack)
  base.next_state_hash = tensor_hash(next_state.stack)
  base.state_shape = tensor_shape(state.stack)
  base.next_state_shape = tensor_shape(next_state.stack)
  base.state_dtype = "uint8"
  base.next_state_dtype = "uint8"
  base.layout = "CHW"
  base.normalized_state_hash = normalized_hash(state.stack)
  base.normalized_next_state_hash = normalized_hash(next_state.stack)
  base.normalized_dtype = "float32"
  base.normalized_range = {0.0, 1.0}
  base.action = action_record.action
  base.action_one_based = action_record.action_one_based
  base.ale_action_code = action_record.ale_action_code
  base.reward = action_record.clipped_reward
  base.raw_reward = action_record.raw_reward
  base.terminal_mask = terminal_record.terminal_mask
  base.life_loss_terminal = terminal_record.life_loss_terminal
  base.true_terminal = terminal_record.true_terminal
  base.next_state_zeroed_or_masked = #next_state.zeroed_positions > 0
  base.action_record_replay_index = action_record.replay_index
  base.terminal_record_replay_index = terminal_record.replay_index
  local state_episode_ids, next_state_episode_ids = {}, {}
  for i = 1, #state.raw_frame_indices do state_episode_ids[i] = records[state.raw_frame_indices[i] + 1].episode_id end
  for i = 1, #next_state.raw_frame_indices do next_state_episode_ids[i] = records[next_state.raw_frame_indices[i] + 1].episode_id end
  base.episode_context = {
    state_episode_ids = state_episode_ids,
    next_state_episode_ids = next_state_episode_ids,
    action_episode_id = action_record.episode_id,
    terminal_episode_id = terminal_record.episode_id,
  }
  return base
end

local function batch_plan(records, requested)
  local valid = {}
  for _, index in ipairs(requested) do
    local accepted = validity(records, index)
    if accepted then valid[#valid + 1] = index end
  end
  local terminal_valid, life_valid, wrap_valid = {}, {}, {}
  local wrap = records[1].wrap_insert_index
  for _, index in ipairs(valid) do
    local sample = sample_at(records, index)
    if sample.terminal_mask then terminal_valid[#terminal_valid + 1] = index end
    if sample.life_loss_terminal then life_valid[#life_valid + 1] = index end
    if math.abs(index - wrap) <= 8 then wrap_valid[#wrap_valid + 1] = index end
  end
  local batches = {}
  if #valid >= 1 then batches[#batches + 1] = {"batch_1", {valid[1]}} end
  if #valid >= 4 then batches[#batches + 1] = {"batch_4", {valid[1], valid[2], valid[3], valid[4]}} end
  if #valid >= 32 then
    local batch = {}
    for i = 1, 32 do batch[i] = valid[i] end
    batches[#batches + 1] = {"batch_32", batch}
  end
  if #terminal_valid > 0 then
    local batch = {}
    for i = 1, math.min(4, #terminal_valid) do batch[i] = terminal_valid[i] end
    batches[#batches + 1] = {"batch_terminal", batch}
  end
  if #life_valid > 0 then
    local batch = {}
    for i = 1, math.min(4, #life_valid) do batch[i] = life_valid[i] end
    batches[#batches + 1] = {"batch_life_loss", batch}
  end
  if #wrap_valid > 0 then
    local batch = {}
    for i = 1, math.min(8, #wrap_valid) do batch[i] = wrap_valid[i] end
    batches[#batches + 1] = {"batch_near_wrap", batch}
  end
  return batches
end

local function batch_hashes(records, indices)
  local state_batch = torch.ByteTensor(#indices, HIST_LEN, 84, 84):zero()
  local next_batch = torch.ByteTensor(#indices, HIST_LEN, 84, 84):zero()
  local samples = {}
  for i, index in ipairs(indices) do
    local sample = sample_at(records, index)
    samples[i] = sample
    state_batch[i]:copy(stack_components(records, index).stack)
    next_batch[i]:copy(stack_components(records, index + 1).stack)
  end
  return state_batch, next_batch, samples
end

local function build_batch(records, indices, name)
  for _, index in ipairs(indices) do
    local accepted = validity(records, index)
    if not accepted then
      return {phase = "replay_batch", batch_name = name, accepted = false, requested_indices = indices, rejection_reason = "batch_contains_rejected_index"}
    end
  end
  local state_batch, next_batch, samples = batch_hashes(records, indices)
  local actions, actions_one_based, rewards, raw_rewards, terminal_masks, life_terms, true_terms, state_hashes, next_hashes = {}, {}, {}, {}, {}, {}, {}, {}, {}
  for i, sample in ipairs(samples) do
    actions[i] = sample.action
    actions_one_based[i] = sample.action_one_based
    rewards[i] = sample.reward
    raw_rewards[i] = sample.raw_reward
    terminal_masks[i] = sample.terminal_mask
    life_terms[i] = sample.life_loss_terminal
    true_terms[i] = sample.true_terminal
    state_hashes[i] = sample.state_hash
    next_hashes[i] = sample.next_state_hash
  end
  return {
    phase = "replay_batch",
    batch_name = name,
    accepted = true,
    requested_indices = indices,
    batch_size = #indices,
    layout = "NCHW",
    state_batch_shape = tensor_shape(state_batch),
    next_state_batch_shape = tensor_shape(next_batch),
    state_batch_dtype = "uint8",
    next_state_batch_dtype = "uint8",
    state_batch_hash = tensor_hash(state_batch),
    next_state_batch_hash = tensor_hash(next_batch),
    normalized_state_batch_hash = normalized_hash(state_batch),
    normalized_next_state_batch_hash = normalized_hash(next_batch),
    normalized_dtype = "float32",
    actions = actions,
    actions_one_based = actions_one_based,
    rewards = rewards,
    raw_rewards = raw_rewards,
    terminal_masks = terminal_masks,
    life_loss_terminals = life_terms,
    true_terminals = true_terms,
    state_hashes = state_hashes,
    next_state_hashes = next_hashes,
  }
end

local records = read_records(REPLAY_DIR .. "/records.tsv")
local requested = read_indices(REQUESTED)
local rows = {}
for position, index in ipairs(requested) do
  local row = sample_at(records, index)
  row.phase = "replay_sample"
  row.source = "deepmind"
  row.step = position - 1
  row.request_position = position - 1
  rows[#rows + 1] = row
end

for _, item in ipairs(batch_plan(records, requested)) do
  local row = build_batch(records, item[2], item[1])
  row.source = "deepmind"
  row.step = "batch:" .. item[1]
  rows[#rows + 1] = row
end

mkdir_p(dirname(OUT))
local out = assert(io.open(OUT, "w"))
for _, row in ipairs(rows) do out:write(encode_json(row)); out:write("\n") end
out:close()
print("wrote " .. OUT)
print("rows: " .. tostring(#rows))
