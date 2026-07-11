#!/usr/bin/env luajit

-- Trace raw DeepMind Torch7/ALE environment facts for Stage 1.
--
-- This file is intentionally a thin adapter around the old alewrap API. The
-- exact constructor and action mapping can vary across local DeepMind forks, so
-- failures here should be treated as API-discovery work rather than audit
-- results.

local ok_torch, torch = pcall(require, "torch")
if not ok_torch then
  error("torch is required for audit/deepmind/trace_env.lua")
end

local function getenv(name, default)
  local value = os.getenv(name)
  if value == nil or value == "" then
    return default
  end
  return value
end

local ROM = getenv("ROM", "/home/uace/dqn/reference/DeepMind-Atari-Deep-Q-Learner/roms/breakout.bin")
local SEED = tonumber(getenv("SEED", "1"))
local TRACE_STEPS = tonumber(getenv("TRACE_STEPS", "200"))
local FRAME_SKIP = tonumber(getenv("FRAME_SKIP", "4"))
local OUT = getenv("OUT", "audit_outputs")
local ACTION_TAPE_TXT = getenv("ACTION_TAPE_TXT", OUT .. "/action_tape_seed" .. SEED .. "_" .. TRACE_STEPS .. ".txt")
local DEEPMIND_ENV_OUT = getenv("DEEPMIND_ENV_OUT", OUT .. "/deepmind_env.jsonl")
local FRAME_DUMP_DIR = getenv("FRAME_DUMP_DIR", OUT .. "/frames")
local FRAME_DUMP_STEPS = tonumber(getenv("FRAME_DUMP_STEPS", "20"))
local FRAME_DUMP_PNG = getenv("FRAME_DUMP_PNG", "1")
local ACTION_TAPE_MODE = getenv("ACTION_TAPE_MODE", "index")
local LUA_ACTION_OFFSET = tonumber(getenv("LUA_ACTION_OFFSET", "1"))
local DM_ACTION_MODE = getenv("DM_ACTION_MODE", "index")

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
  if type(tbl) ~= "table" then
    return false
  end
  local count = 0
  local max_key = 0
  for key, _ in pairs(tbl) do
    if type(key) ~= "number" then
      return false
    end
    if key > max_key then
      max_key = key
    end
    count = count + 1
  end
  return max_key == count
end

local function sorted_keys(tbl)
  local keys = {}
  for key, _ in pairs(tbl) do
    table.insert(keys, key)
  end
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
    if value == JSON_NULL then
      return "null"
    end
    if is_array(value) then
      local parts = {}
      for i = 1, #value do
        table.insert(parts, encode_json(value[i]))
      end
      return "[" .. table.concat(parts, ",") .. "]"
    end
    local parts = {}
    for _, key in ipairs(sorted_keys(value)) do
      if value[key] ~= nil then
        table.insert(parts, '"' .. json_escape(key) .. '":' .. encode_json(value[key]))
      end
    end
    return "{" .. table.concat(parts, ",") .. "}"
  end
  return '"' .. json_escape(tostring(value)) .. '"'
end

local function read_action_tape(path)
  local actions = {}
  local handle = assert(io.open(path, "r"))
  for line in handle:lines() do
    local stripped = line:match("^%s*(.-)%s*$")
    if stripped ~= "" then
      table.insert(actions, tonumber(stripped))
    end
  end
  handle:close()
  return actions
end

local function tensor_shape(tensor)
  if tensor == nil or torch.isTensor(tensor) == false then
    return nil
  end
  local shape = {}
  for dim = 1, tensor:dim() do
    table.insert(shape, tensor:size(dim))
  end
  return shape
end

local function normalize_frame_tensor(tensor)
  if tensor == nil or torch.isTensor(tensor) == false then
    return tensor
  end
  -- DeepMind alewrap returns RGB frames as C x H x W. Python Gymnasium emits
  -- H x W x C. Compare the same byte order and shape at the audit boundary.
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
  if normalized == nil or torch.isTensor(normalized) == false then
    return normalized
  end
  if torch.type(normalized) == "torch.ByteTensor" then
    return normalized:contiguous()
  end
  local numeric = normalized:double()
  if numeric:nElement() > 0 and tonumber(numeric:min()) >= 0 and tonumber(numeric:max()) <= 1 then
    numeric:mul(255)
  end
  numeric:clamp(0, 255)
  return numeric:add(0.5):floor():byte():contiguous()
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
    return {
      hash = nil,
      shape = nil,
      dtype = type(tensor),
    }
  end
  local normalized = frame_to_byte_hwc(tensor)
  local numeric = normalized:double()
  local mean = tonumber(numeric:mean())
  local centered = numeric:clone():add(-mean)
  local population_std = math.sqrt(tonumber(centered:pow(2):mean()))
  local stats = {
    hash = sha256_bytes(tensor_bytes(normalized)),
    shape = tensor_shape(normalized),
    dtype = torch.type(tensor),
    min = tonumber(numeric:min()),
    max = tonumber(numeric:max()),
    mean = mean,
    std = population_std,
  }
  if normalized:dim() == 3 and (normalized:size(3) == 1 or normalized:size(3) == 3 or normalized:size(3) == 4) then
    local channel_means = {}
    local channel_mins = {}
    local channel_maxs = {}
    local channel_hashes = {}
    for c = 1, normalized:size(3) do
      local channel = normalized[{{}, {}, c}]:contiguous()
      local channel_numeric = channel:double()
      channel_means[c] = tonumber(channel_numeric:mean())
      channel_mins[c] = tonumber(channel_numeric:min())
      channel_maxs[c] = tonumber(channel_numeric:max())
      channel_hashes[c] = sha256_bytes(tensor_bytes(channel))
    end
    stats.channel_means = channel_means
    stats.channel_mins = channel_mins
    stats.channel_maxs = channel_maxs
    stats.channel_hashes = channel_hashes
  end
  return stats
end

local function tensor_byte_values(tensor)
  if tensor == nil or torch.isTensor(tensor) == false then
    return nil
  end
  local bytes = {}
  local flat = frame_to_byte_hwc(tensor):view(-1)
  for i = 1, flat:nElement() do
    bytes[i] = math.floor(tonumber(flat[i]) + 0.5) % 256
  end
  return bytes
end

local function ram_stats(tensor)
  if tensor == nil or torch.isTensor(tensor) == false then
    return JSON_NULL
  end
  local stats = tensor_stats(tensor)
  stats.bytes = tensor_byte_values(tensor)
  return stats
end

local function write_npy_uint8(path, tensor)
  local frame = frame_to_byte_hwc(tensor)
  if frame == nil or torch.isTensor(frame) == false then
    return
  end
  mkdir_p(dirname(path))
  local shape_parts = {}
  for dim = 1, frame:dim() do
    shape_parts[#shape_parts + 1] = tostring(frame:size(dim))
  end
  if frame:dim() == 1 then
    shape_parts[1] = shape_parts[1] .. ","
  end
  local header = "{'descr': '|u1', 'fortran_order': False, 'shape': (" .. table.concat(shape_parts, ", ") .. "), }"
  local magic_len = 10
  local padding = 16 - ((magic_len + #header + 1) % 16)
  if padding == 16 then
    padding = 0
  end
  header = header .. string.rep(" ", padding) .. "\n"
  local header_len = #header
  local handle = assert(io.open(path, "wb"))
  handle:write(string.char(0x93) .. "NUMPY")
  handle:write(string.char(1, 0))
  handle:write(string.char(header_len % 256, math.floor(header_len / 256)))
  handle:write(header)
  handle:write(tensor_bytes(frame))
  handle:close()
end

local function write_png_if_possible(path, tensor)
  local ok_image, image = pcall(require, "image")
  if not ok_image then
    return
  end
  local frame = frame_to_byte_hwc(tensor)
  if frame == nil or torch.isTensor(frame) == false then
    return
  end
  local out = frame
  if frame:dim() == 3 and frame:size(3) == 3 then
    out = frame:transpose(2, 3):transpose(1, 2):double():div(255)
  elseif frame:dim() == 3 and frame:size(3) == 1 then
    out = frame[{{}, {}, 1}]:double():div(255)
  else
    out = frame:double():div(255)
  end
  local ok = pcall(function() image.save(path, out) end)
  if not ok then
    return
  end
end

local function dump_frame(tensor, name)
  if FRAME_DUMP_DIR == nil or FRAME_DUMP_DIR == "" then
    return
  end
  write_npy_uint8(FRAME_DUMP_DIR .. "/" .. name .. ".npy", tensor)
  if FRAME_DUMP_PNG ~= "0" then
    write_png_if_possible(FRAME_DUMP_DIR .. "/" .. name .. ".png", tensor)
  end
end

local function call_method(obj, names, ...)
  for _, name in ipairs(names) do
    if obj ~= nil and obj[name] ~= nil then
      local args = {...}
      local ok, result = pcall(function() return obj[name](obj, unpack(args)) end)
      if ok then
        return result
      end
    end
  end
  return nil
end

local function make_env()
  local ok_ale, alewrap = pcall(require, "alewrap")
  if not ok_ale then
    error("could not require alewrap; run from the DeepMind Torch7 environment")
  end

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
    if ok and env ~= nil then
      return env
    end
    last_error = env
  end
  error("could not construct alewrap.GameEnvironment: " .. tostring(last_error))
end

local function get_screen(env)
  return call_method(env, {"getScreen", "get_screen", "screen", "getState"})
end

local function get_lives(env)
  if env ~= nil and env._state ~= nil and env._state.lives ~= nil then
    return env._state.lives
  end
  if env ~= nil and env.env ~= nil and env.env.lives ~= nil then
    local ok, lives = pcall(function() return env.env:lives() end)
    if ok then
      return lives
    end
  end
  return call_method(env, {"lives", "getLives", "ale_lives"})
end

local function get_frame_number(env)
  return call_method(env, {"getFrameNumber", "frameNumber", "getFrame"})
end

local function get_score(env)
  if env ~= nil and env._state ~= nil and env._state.score ~= nil then
    return env._state.score
  end
  return call_method(env, {"getEpisodeScore", "getScore", "score"})
end

local function get_ram(env)
  local ram = call_method(env, {"getRAM", "getRam", "ram"})
  if ram ~= nil then
    return ram
  end
  if env ~= nil and env.env ~= nil then
    return call_method(env.env, {"getRAM", "getRam", "ram"})
  end
  if env ~= nil and env.ale ~= nil then
    return call_method(env.ale, {"getRAM", "getRam", "ram"})
  end
  return nil
end

local function get_done(env)
  if env ~= nil and env._state ~= nil and env._state.terminal ~= nil then
    return env._state.terminal and true or false
  end
  local done = call_method(env, {"game_over", "gameOver", "isTerminal", "terminal"})
  if done == nil then
    return false
  end
  return done and true or false
end

local function reset_env(env)
  if env ~= nil and env.reset ~= nil then
    local ok = pcall(function() return env:reset() end)
    if ok then
      return
    end
  end
  call_method(env, {"reset_game", "newGame"})
end

local function step_env(env, action)
  if env ~= nil and env._step ~= nil then
    local frame, reward, terminal, lives = env:_step(action)
    env:_updateState(frame, reward, terminal, lives)
    return frame, reward, terminal, lives
  end
  if env ~= nil and env.play ~= nil then
    local result = env:play(action)
    return result.data, result.reward, result.terminal, result.lives
  end
  if env ~= nil and env.step ~= nil then
    local frame, reward, terminal = env:step(action, false)
    return frame, reward, terminal, get_lives(env)
  end
  if env ~= nil and env.act ~= nil then
    local reward = env:act(action)
    return get_screen(env), reward, get_done(env), get_lives(env)
  end
  error("could not find a supported DeepMind env stepping API")
end

local function get_actions(env)
  local actions = call_method(env, {"getActions", "actions", "getLegalActionSet"})
  if actions == nil then
    return nil
  end
  if torch.isTensor(actions) then
    local result = {}
    for i = 1, actions:nElement() do
      result[i] = tonumber(actions:view(-1)[i])
    end
    return result
  end
  return actions
end

local function action_meaning_for(action_index, valid_actions)
  local meanings = {"NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"}
  if ACTION_TAPE_MODE == "index" and valid_actions ~= nil then
    local code = valid_actions[action_index + 1]
    if code ~= nil then
      return meanings[code + 1]
    end
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
  if a == nil then
    return b
  end
  if b == nil then
    return a
  end
  local out = a:clone()
  local flat_out = out:view(out:nElement())
  local flat_b = b:view(b:nElement())
  for i = 1, flat_out:nElement() do
    if flat_b[i] > flat_out[i] then
      flat_out[i] = flat_b[i]
    end
  end
  return out
end

local function write_jsonl(path, rows)
  mkdir_p(dirname(path))
  local handle = assert(io.open(path, "w"))
  for _, row in ipairs(rows) do
    handle:write(encode_json(row))
    handle:write("\n")
  end
  handle:close()
end

local actions = read_action_tape(ACTION_TAPE_TXT)
local env = make_env()
local valid_actions = get_actions(env)

reset_env(env)
local reset_screen = get_screen(env)
dump_frame(reset_screen, "deepmind_init")

local rows = {
  {
    phase = "init",
    source = "deepmind",
    rom = ROM,
    seed = SEED,
    frame_skip = FRAME_SKIP,
    action_tape_mode = ACTION_TAPE_MODE,
    gymnasium_version = JSON_NULL,
    ale_py_version = JSON_NULL,
    alewrap_version = JSON_NULL,
    ale_version = JSON_NULL,
    action_space_n = valid_actions and #valid_actions or nil,
    action_values = valid_actions,
    legal_action_set = valid_actions,
    minimal_action_set = valid_actions,
    action_meanings = JSON_NULL,
    dm_action_mode = DM_ACTION_MODE,
    lua_action_offset = LUA_ACTION_OFFSET,
    ale_lives = get_lives(env),
    ale_frame_number = get_frame_number(env) or JSON_NULL,
    frame_number = get_frame_number(env) or JSON_NULL,
    score = get_score(env) or JSON_NULL,
    ram = ram_stats(get_ram(env)),
    raw_frame = tensor_stats(reset_screen),
    reset_info = {},
  }
}

for step = 0, math.min(TRACE_STEPS, #actions) - 1 do
  if get_done(env) then
    table.insert(rows, {phase = "trace_end", source = "deepmind", step = step, reason = "done"})
    break
  end

  local tape_action = actions[step + 1]
  local lua_action, action_index_used, ale_action, action_meaning = map_action(tape_action, valid_actions)
  local lives_before = get_lives(env)
  if step <= FRAME_DUMP_STEPS then
    dump_frame(get_screen(env), string.format("deepmind_step_%03d_pre", step))
  end
  local frames = {}
  local per_frame_rewards = {}
  local repeats = {}
  local total_reward = 0

  for repeat_index = 1, FRAME_SKIP do
    local frame, reward, terminal, lives = step_env(env, lua_action)
    total_reward = total_reward + tonumber(reward)
    table.insert(per_frame_rewards, tonumber(reward))
    table.insert(frames, frame)
    if step <= FRAME_DUMP_STEPS then
      dump_frame(frame, string.format("deepmind_step_%03d_repeat_%03d", step, repeat_index - 1))
    end
    table.insert(repeats, {
      repeat_i = repeat_index - 1,
      tape_action_raw = tape_action,
      action_index_used = action_index_used,
      ale_action_code_used = ale_action,
      action_meaning_used = action_meaning,
      reward = tonumber(reward),
      lives_after = lives,
      terminal = terminal and true or false,
      truncated = false,
      frame_number = get_frame_number(env) or JSON_NULL,
      score = get_score(env) or JSON_NULL,
      ram = ram_stats(get_ram(env)),
      raw_frame = tensor_stats(frame),
    })
    if terminal then
      break
    end
  end

  local pooled = nil
  local pooling_frame_indices = {}
  if #frames >= 2 then
    pooled = max_pool(frames[#frames - 1], frames[#frames])
    pooling_frame_indices = {#frames - 2, #frames - 1}
  elseif #frames == 1 then
    pooled = frames[1]
    pooling_frame_indices = {0}
  end

  if step <= FRAME_DUMP_STEPS then
    dump_frame(frames[#frames], string.format("deepmind_step_%03d", step))
    if #frames >= 2 then
      dump_frame(frames[#frames - 1], string.format("deepmind_step_%03d_pool_src_%03d", step, #frames - 2))
      dump_frame(frames[#frames], string.format("deepmind_step_%03d_pool_src_%03d", step, #frames - 1))
    end
    dump_frame(pooled, string.format("deepmind_step_%03d_pooled", step))
  end

  table.insert(rows, {
    phase = "agent_step",
    source = "deepmind",
    step = step,
    action = tape_action,
    tape_action_raw = tape_action,
    action_index_used = action_index_used,
    ale_action_code_used = ale_action,
    action_meaning_used = action_meaning,
    legal_action_set = valid_actions,
    minimal_action_set = valid_actions,
    lua_action = lua_action,
    ale_action = ale_action,
    repeat_count = #frames,
    repeats = repeats,
    per_frame_rewards = per_frame_rewards,
    reward = total_reward,
    lives_before = lives_before,
    lives_after = get_lives(env),
    terminated = get_done(env),
    truncated = false,
    done = get_done(env),
    ale_frame_number = get_frame_number(env) or JSON_NULL,
    frame_number = get_frame_number(env) or JSON_NULL,
    score = get_score(env) or JSON_NULL,
    ram = ram_stats(get_ram(env)),
    pooling_frame_indices = pooling_frame_indices,
    raw_frame = tensor_stats(frames[#frames]),
    pooled_frame = tensor_stats(pooled),
    info = {},
  })
end

write_jsonl(DEEPMIND_ENV_OUT, rows)
print("wrote " .. DEEPMIND_ENV_OUT)
