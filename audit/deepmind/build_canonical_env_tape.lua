#!/usr/bin/env luajit

-- Build a frozen DeepMind ALE tape for Branch 2 audits.
--
-- This script is DeepMind-only: it runs the old alewrap environment once,
-- writes raw uint8 HWC .npy frames for Python consumers, writes matching CHW
-- ByteTensor .t7 frames for Torch7 consumers, and records transition metadata.

local ok_torch, torch = pcall(require, "torch")
if not ok_torch then
  error("torch is required")
end

local function getenv(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then return default end
  return value
end

local ROM = getenv("ROM", "/home/uace/dqn/reference/DeepMind-Atari-Deep-Q-Learner/roms/breakout.bin")
local SEED = tonumber(getenv("SEED", "1"))
local TRACE_STEPS = tonumber(getenv("TRACE_STEPS", "64"))
local FRAME_SKIP = tonumber(getenv("FRAME_SKIP", "4"))
local ACTION_TAPE_TXT = getenv("ACTION_TAPE_TXT", "action_tape.txt")
local TAPE_DIR = getenv("CANONICAL_TAPE_DIR", getenv("OUT", "audit_outputs") .. "/canonical_frames")
local ACTION_TAPE_MODE = getenv("ACTION_TAPE_MODE", "index")
local DM_ACTION_MODE = getenv("DM_ACTION_MODE", "ale")
local LUA_ACTION_OFFSET = tonumber(getenv("LUA_ACTION_OFFSET", "1"))

torch.manualSeed(SEED)

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

local function basename_without_bin(path)
  local name = tostring(path):match("([^/]+)$") or tostring(path)
  return (name:gsub("%.bin$", ""))
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

local function write_json(path, value)
  mkdir_p(dirname(path))
  local handle = assert(io.open(path, "w"))
  handle:write(encode_json(value))
  handle:write("\n")
  handle:close()
end

local function append_jsonl(handle, value)
  handle:write(encode_json(value))
  handle:write("\n")
end

local function read_action_tape(path)
  local actions = {}
  local handle = assert(io.open(path, "r"))
  for line in handle:lines() do
    local stripped = line:match("^%s*(.-)%s*$")
    if stripped ~= "" then actions[#actions + 1] = tonumber(stripped) end
  end
  handle:close()
  return actions
end

local function tensor_shape(tensor)
  if tensor == nil or torch.isTensor(tensor) == false then return nil end
  local shape = {}
  for dim = 1, tensor:dim() do shape[#shape + 1] = tensor:size(dim) end
  return shape
end

local function normalize_frame_tensor(tensor)
  if tensor == nil or torch.isTensor(tensor) == false then return tensor end
  if tensor:dim() == 4 and tensor:size(1) == 1 and tensor:size(2) == 3 then
    local frame = tensor[1]
    return frame:transpose(1, 2):transpose(2, 3):contiguous()
  elseif tensor:dim() == 3 and tensor:size(1) == 3 then
    return tensor:transpose(1, 2):transpose(2, 3):contiguous()
  end
  return tensor:contiguous()
end

local function frame_to_byte_hwc(tensor)
  local normalized = normalize_frame_tensor(tensor)
  if normalized == nil or torch.isTensor(normalized) == false then return normalized end
  if torch.type(normalized) == "torch.ByteTensor" then return normalized:contiguous() end
  local numeric = normalized:double()
  if numeric:nElement() > 0 and tonumber(numeric:min()) >= 0 and tonumber(numeric:max()) <= 1 then
    numeric:mul(255)
  end
  numeric:clamp(0, 255)
  return numeric:add(0.5):floor():byte():contiguous()
end

local function frame_to_byte_chw(tensor)
  local hwc = frame_to_byte_hwc(tensor)
  if hwc:dim() == 3 and hwc:size(3) == 3 then
    return hwc:transpose(2, 3):transpose(1, 2):contiguous()
  end
  return hwc:contiguous()
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

local function tensor_stats(tensor)
  if tensor == nil or torch.isTensor(tensor) == false then
    return {hash = JSON_NULL, shape = JSON_NULL, dtype = type(tensor)}
  end
  local normalized = frame_to_byte_hwc(tensor)
  local numeric = normalized:double()
  return {
    hash = sha256_bytes(tensor_bytes(normalized)),
    shape = tensor_shape(normalized),
    dtype = torch.type(tensor),
    min = tonumber(numeric:min()),
    max = tonumber(numeric:max()),
    mean = tonumber(numeric:mean()),
    std = math.sqrt(tonumber(numeric:clone():add(-tonumber(numeric:mean())):pow(2):mean())),
  }
end

local function write_npy_uint8(path, tensor)
  local frame = frame_to_byte_hwc(tensor)
  if frame == nil or torch.isTensor(frame) == false then return end
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

local function save_frame_pair(npy_path, t7_path, tensor)
  write_npy_uint8(npy_path, tensor)
  torch.save(t7_path, frame_to_byte_chw(tensor))
  return tensor_stats(tensor)
end

local function call_method(obj, names, ...)
  for _, name in ipairs(names) do
    if obj ~= nil and obj[name] ~= nil then
      local args = {...}
      local ok, result = pcall(function() return obj[name](obj, unpack(args)) end)
      if ok then return result end
    end
  end
  return nil
end

local function make_env()
  local ok_ale, alewrap = pcall(require, "alewrap")
  if not ok_ale then error("could not require alewrap") end
  local game_name = getenv("DM_GAME", basename_without_bin(ROM))
  local game_path = getenv("DM_GAME_PATH", dirname(ROM))
  local attempts = {
    function()
      return alewrap.GameEnvironment({
        game_path = game_path,
        env = game_name,
        env_params = {useRGB = true},
        actrep = 1,
        random_starts = 1,
        verbose = 0,
      })
    end,
    function() return alewrap.game(game_name, {useRGB = true}, game_path) end,
  }
  local last_error = nil
  for _, attempt in ipairs(attempts) do
    local ok, env = pcall(attempt)
    if ok and env ~= nil then return env end
    last_error = env
  end
  error("could not construct alewrap.GameEnvironment: " .. tostring(last_error))
end

local function get_screen(env)
  return call_method(env, {"getScreen", "get_screen", "screen", "getState"})
end

local function get_lives(env)
  if env ~= nil and env._state ~= nil and env._state.lives ~= nil then return env._state.lives end
  if env ~= nil and env.env ~= nil and env.env.lives ~= nil then
    local ok, lives = pcall(function() return env.env:lives() end)
    if ok then return lives end
  end
  return call_method(env, {"lives", "getLives", "ale_lives"})
end

local function get_done(env)
  if env ~= nil and env._state ~= nil and env._state.terminal ~= nil then
    return env._state.terminal and true or false
  end
  local done = call_method(env, {"game_over", "gameOver", "isTerminal", "terminal"})
  if done == nil then return false end
  return done and true or false
end

local function reset_env(env)
  if env ~= nil and env.reset ~= nil then
    local ok = pcall(function() return env:reset() end)
    if ok then return end
  end
  call_method(env, {"reset_game", "newGame"})
end

local function step_env(env, action)
  if env ~= nil and env._step ~= nil then
    local frame, reward, terminal, lives = env:_step(action)
    env:_updateState(frame, reward, terminal, lives)
    return frame, reward, terminal, lives
  end
  if env ~= nil and env.step ~= nil then
    local frame, reward, terminal = env:step(action, false)
    return frame, reward, terminal, get_lives(env)
  end
  error("could not find supported DeepMind env stepping API")
end

local function get_actions(env)
  local actions = call_method(env, {"getActions", "actions", "getLegalActionSet"})
  if actions == nil then return nil end
  if torch.isTensor(actions) then
    local result = {}
    for i = 1, actions:nElement() do result[i] = tonumber(actions:view(-1)[i]) end
    return result
  end
  return actions
end

local function action_meaning_for(action_index, valid_actions)
  local meanings = {"NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"}
  if ACTION_TAPE_MODE == "index" and valid_actions ~= nil then
    local code = valid_actions[action_index + 1]
    if code ~= nil then return meanings[code + 1] end
  end
  return meanings[action_index + 1]
end

local function map_action(tape_action, valid_actions)
  if ACTION_TAPE_MODE == "index" then
    if valid_actions ~= nil then
      return valid_actions[tape_action + 1], tape_action, valid_actions[tape_action + 1], action_meaning_for(tape_action, valid_actions)
    end
    return tape_action + LUA_ACTION_OFFSET, tape_action, tape_action + LUA_ACTION_OFFSET, action_meaning_for(tape_action, valid_actions)
  elseif ACTION_TAPE_MODE == "ale_code" then
    return tape_action, tape_action, tape_action, action_meaning_for(tape_action, nil)
  elseif DM_ACTION_MODE == "ale" and valid_actions ~= nil then
    return valid_actions[tape_action + 1], tape_action, valid_actions[tape_action + 1], action_meaning_for(tape_action, valid_actions)
  end
  return tape_action + LUA_ACTION_OFFSET, tape_action, tape_action + LUA_ACTION_OFFSET, action_meaning_for(tape_action, valid_actions)
end

local function max_pool(a, b)
  if a == nil then return b end
  if b == nil then return a end
  local out = a:clone()
  local flat_out = out:view(out:nElement())
  local flat_b = b:view(b:nElement())
  for i = 1, flat_out:nElement() do
    if flat_b[i] > flat_out[i] then flat_out[i] = flat_b[i] end
  end
  return out
end

local function clipped_reward(reward)
  if reward > 1 then return 1 end
  if reward < -1 then return -1 end
  return reward
end

mkdir_p(TAPE_DIR .. "/frames")
mkdir_p(TAPE_DIR .. "/frames_t7")
mkdir_p(TAPE_DIR .. "/pooled")
mkdir_p(TAPE_DIR .. "/pooled_t7")

local actions = read_action_tape(ACTION_TAPE_TXT)
local env = make_env()
local valid_actions = get_actions(env)
reset_env(env)

local transitions_path = TAPE_DIR .. "/transitions.jsonl"
local manifest_jsonl_path = TAPE_DIR .. "/manifest.jsonl"
local pooled_paths_path = TAPE_DIR .. "/pooled_paths.txt"
local pooled_t7_paths_path = TAPE_DIR .. "/pooled_t7_paths.txt"
local transitions = assert(io.open(transitions_path, "w"))
local manifest_jsonl = assert(io.open(manifest_jsonl_path, "w"))
local pooled_paths = assert(io.open(pooled_paths_path, "w"))
local pooled_t7_paths = assert(io.open(pooled_t7_paths_path, "w"))

local frame_counter = 0
local function next_frame_paths()
  local id = frame_counter
  frame_counter = frame_counter + 1
  return id,
    string.format("%s/frames/frame_%06d.npy", TAPE_DIR, id),
    string.format("%s/frames_t7/frame_%06d.t7", TAPE_DIR, id)
end

local manifest = {
  phase = "freeze_manifest",
  canonical_source = "deepmind_alewrap",
  game = basename_without_bin(ROM),
  rom = ROM,
  seed = SEED,
  frame_skip = FRAME_SKIP,
  trace_steps = TRACE_STEPS,
  action_tape_txt = ACTION_TAPE_TXT,
  action_tape_mode = ACTION_TAPE_MODE,
  action_values = valid_actions,
  transitions_path = transitions_path,
  pooled_paths_path = pooled_paths_path,
  pooled_t7_paths_path = pooled_t7_paths_path,
}
write_json(TAPE_DIR .. "/manifest.json", manifest)
append_jsonl(manifest_jsonl, manifest)

for step = 0, math.min(TRACE_STEPS, #actions) - 1 do
  if get_done(env) then
    append_jsonl(transitions, {phase = "trace_end", step = step, reason = "done"})
    break
  end

  local pre = get_screen(env)
  local pre_id, pre_path, pre_t7_path = next_frame_paths()
  local pre_stats = save_frame_pair(pre_path, pre_t7_path, pre)
  local tape_action = actions[step + 1]
  local lua_action, action_index, ale_action, action_meaning = map_action(tape_action, valid_actions)
  local lives_before = get_lives(env)
  local frames = {}
  local repeat_rows = {}
  local total_reward = 0
  local terminal = false
  local lives_after = lives_before

  for repeat_i = 0, FRAME_SKIP - 1 do
    local frame, reward, repeat_terminal, lives = step_env(env, lua_action)
    frames[#frames + 1] = frame
    total_reward = total_reward + tonumber(reward)
    terminal = repeat_terminal and true or false
    lives_after = lives
    local frame_id, frame_path, frame_t7_path = next_frame_paths()
    local stats = save_frame_pair(frame_path, frame_t7_path, frame)
    repeat_rows[#repeat_rows + 1] = {
      repeat_i = repeat_i,
      frame_id = frame_id,
      frame_path = frame_path,
      frame_t7_path = frame_t7_path,
      reward = tonumber(reward),
      lives_after = lives,
      terminal = terminal,
      raw_frame = stats,
    }
    if terminal then break end
  end

  local pooled
  local pooling_frame_indices
  if #frames >= 2 then
    pooled = max_pool(frames[#frames - 1], frames[#frames])
    pooling_frame_indices = {#frames - 2, #frames - 1}
  else
    pooled = frames[#frames]
    pooling_frame_indices = {#frames - 1}
  end
  local pooled_path = string.format("%s/pooled/pooled_%06d.npy", TAPE_DIR, step)
  local pooled_t7_path = string.format("%s/pooled_t7/pooled_%06d.t7", TAPE_DIR, step)
  local pooled_stats = save_frame_pair(pooled_path, pooled_t7_path, pooled)
  pooled_paths:write(step .. "\t" .. pooled_path .. "\n")
  pooled_t7_paths:write(step .. "\t" .. pooled_t7_path .. "\n")

  local row = {
    phase = "transition",
    step = step,
    pre_frame_id = pre_id,
    pre_frame_path = pre_path,
    pre_frame_t7_path = pre_t7_path,
    pre_frame = pre_stats,
    tape_action_raw = tape_action,
    action_index = action_index,
    ale_action_code = ale_action,
    action_meaning = action_meaning,
    reward = total_reward,
    clipped_reward = clipped_reward(total_reward),
    lives_before = lives_before,
    lives_after = lives_after,
    life_loss = (lives_before ~= nil and lives_after ~= nil and lives_after < lives_before) or false,
    terminal = terminal,
    repeat_count = #frames,
    repeats = repeat_rows,
    pooling_frame_indices = pooling_frame_indices,
    pooled_path = pooled_path,
    pooled_t7_path = pooled_t7_path,
    pooled_frame = pooled_stats,
  }
  append_jsonl(transitions, row)
  append_jsonl(manifest_jsonl, row)
end

transitions:close()
manifest_jsonl:close()
pooled_paths:close()
pooled_t7_paths:close()

print("wrote canonical DeepMind env tape: " .. TAPE_DIR)
