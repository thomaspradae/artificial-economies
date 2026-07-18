#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/freeze_config.sh"

STAGE2D_OUT="${STAGE2D_OUT:-$OUT}"
TORCH_DIR="${TORCH_DIR:-$DM_DIR/torch}"
if [[ ! -d "$TORCH_DIR" && -n "${LUAJIT_BIN:-}" ]]; then
  TORCH_DIR="$(cd "$(dirname "$LUAJIT_BIN")/.." && pwd)"
fi

REPORT="$STAGE2D_OUT/source_report.txt"
SNIPPET_DIR="$STAGE2D_OUT/source_snippets"
mkdir -p "$SNIPPET_DIR"

if [[ ! -x "$LUAJIT_BIN" ]]; then
  echo "missing executable LuaJIT: $LUAJIT_BIN" >&2
  exit 2
fi
if [[ ! -d "$DM_DIR" ]]; then
  echo "missing DeepMind dir: $DM_DIR" >&2
  exit 2
fi
if [[ ! -d "$TORCH_DIR" ]]; then
  echo "missing Torch dir: $TORCH_DIR" >&2
  exit 2
fi

copy_snippet() {
  local path="$1"
  if [[ -f "$path" ]]; then
    local safe
    safe="$(printf '%s' "$path" | sed 's#^/##; s#[^A-Za-z0-9._-]#_#g')"
    cp "$path" "$SNIPPET_DIR/$safe"
  fi
}

LUA_INSPECT="$STAGE2D_OUT/inspect_image_package.lua"
cat > "$LUA_INSPECT" <<'LUA'
local function safe_require(name)
  local ok, module = pcall(require, name)
  print("require." .. name .. ".ok=" .. tostring(ok))
  if not ok then print("require." .. name .. ".error=" .. tostring(module)) end
  return ok and module or nil
end

local function describe_function(name, fn)
  print("function." .. name .. ".type=" .. type(fn))
  if type(fn) ~= "function" then return end
  local ok, info = pcall(debug.getinfo, fn, "Snl")
  if not ok then
    print("function." .. name .. ".getinfo_error=" .. tostring(info))
    return
  end
  for _, key in ipairs({"source", "short_src", "what", "linedefined", "lastlinedefined"}) do
    print("function." .. name .. "." .. key .. "=" .. tostring(info[key]))
  end
end

print("_VERSION=" .. tostring(_VERSION))
print("arg0=" .. tostring(arg and arg[0]))
print("package.path=" .. tostring(package.path))
print("package.cpath=" .. tostring(package.cpath))

local torch = safe_require("torch")
local image = safe_require("image")
if torch then
  print("torch.type=" .. tostring(type(torch)))
  print("torch.version=" .. tostring(torch.version))
end
if image then
  print("image.type=" .. tostring(type(image)))
  describe_function("image.scale", image.scale)
  describe_function("image.scaleBilinear", image.scaleBilinear)
  describe_function("image.rgb2y", image.rgb2y)
  local keys = {}
  for key, value in pairs(image) do
    if tostring(key):lower():find("scale") then
      keys[#keys + 1] = tostring(key) .. ":" .. type(value)
    end
  end
  table.sort(keys)
  print("image.scale_keys=" .. table.concat(keys, ","))
  local x = torch and torch.FloatTensor(1, 210, 160):zero()
  if x then
    local ok, y = pcall(function() return image.scale(x, 84, 84, "bilinear") end)
    print("image.scale.float01_smoke.ok=" .. tostring(ok))
    if ok then
      print("image.scale.float01_smoke.type=" .. tostring(torch.type(y)))
      print("image.scale.float01_smoke.dim=" .. tostring(y:dim()))
    else
      print("image.scale.float01_smoke.error=" .. tostring(y))
    end
  end
end
LUA

"$LUAJIT_BIN" "$LUA_INSPECT" > "$STAGE2D_OUT/lua_introspection.txt" 2>&1 || true

DEEP_GREP="$STAGE2D_OUT/grep_deepmind_resize.txt"
TORCH_GREP="$STAGE2D_OUT/grep_torch_resize.txt"
FIND_REPORT="$STAGE2D_OUT/find_image_scale_files.txt"
LIB_REPORT="$STAGE2D_OUT/libimage_shared_objects.txt"
SOURCE_REPORT="$STAGE2D_OUT/native_source_candidates.txt"
ROCKSPEC_REPORT="$STAGE2D_OUT/image_rock_metadata.txt"

grep -RIn "net_downsample_2x_full_y" "$DM_DIR" > "$DEEP_GREP" 2>&1 || true
grep -RIn "scaleBilinear\|image.scale\|bilinear\|rgb2y" "$TORCH_DIR" > "$TORCH_GREP" 2>&1 || true
find "$TORCH_DIR" \( -iname '*image*' -o -iname '*scale*' \) -print > "$FIND_REPORT" 2>&1 || true
find "$TORCH_DIR" -type f \( -name '*image*.so' -o -name 'libimage*' -o -name 'image.so' \) -print > "$LIB_REPORT" 2>&1 || true
find "$TORCH_DIR" -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -print > "$SOURCE_REPORT" 2>&1 || true
find "$TORCH_DIR/lib/luarocks/rocks/image" -maxdepth 3 -type f \( -name '*.rockspec' -o -name 'rock_manifest' -o -name 'manifest' \) -print > "$ROCKSPEC_REPORT" 2>&1 || true

copy_snippet "$DM_DIR/dqn/net_downsample_2x_full_y.lua"
copy_snippet "$DM_DIR/dqn/Scale.lua"
copy_snippet "$TORCH_DIR/share/lua/5.1/image/init.lua"
copy_snippet "$TORCH_DIR/share/lua/5.1/image/test.lua"
while IFS= read -r rock_path; do
  copy_snippet "$rock_path"
done < "$ROCKSPEC_REPORT"

while IFS= read -r source_path; do
  if [[ -f "$source_path" ]] && grep -qE "scaleBilinear|bilinear|rgb2y" "$source_path"; then
    copy_snippet "$source_path"
  fi
done < "$SOURCE_REPORT"

{
  echo "Stage 2d Torch7 resize source report"
  echo
  echo "DM_DIR=$DM_DIR"
  echo "TORCH_DIR=$TORCH_DIR"
  echo "LUAJIT_BIN=$LUAJIT_BIN"
  echo "AUDIT_DIR=$AUDIT_DIR"
  echo "STAGE2D_OUT=$STAGE2D_OUT"
  echo
  echo "LuaJIT version"
  "$LUAJIT_BIN" -v 2>&1 || true
  echo
  echo "Lua introspection"
  cat "$STAGE2D_OUT/lua_introspection.txt"
  echo
  echo "Expected DeepMind call chain"
  echo "net_downsample_2x_full_y.lua -> nn.Scale(84,84,true) -> Scale.lua -> image.rgb2y -> image.scale(..., 'bilinear') -> native image scaleBilinear"
  echo
  echo "DeepMind resize grep"
  cat "$DEEP_GREP"
  echo
  echo "Torch image/resize grep"
  cat "$TORCH_GREP"
  echo
  echo "Image/scale file search"
  cat "$FIND_REPORT"
  echo
  echo "Shared objects"
  cat "$LIB_REPORT"
  echo
  while IFS= read -r lib; do
    if [[ -f "$lib" ]]; then
      echo "ldd $lib"
      ldd "$lib" 2>&1 || true
      echo
      echo "nm -D $lib | grep scale"
      nm -D "$lib" 2>/dev/null | grep -Ei "scale|bilinear|rgb2y" || true
      echo
      echo "nm -a $lib | grep scale"
      nm -a "$lib" 2>/dev/null | grep -Ei "scale|bilinear|rgb2y" || true
      echo
      echo "strings $lib | grep scale"
      strings "$lib" 2>/dev/null | grep -Ei "scaleBilinear|scaleBicubic|scaleSimple|rgb2y" || true
      echo
    fi
  done < "$LIB_REPORT"
  echo "Image rock metadata"
  cat "$ROCKSPEC_REPORT"
  while IFS= read -r rock_path; do
    if [[ -f "$rock_path" ]]; then
      echo "=== $rock_path ==="
      sed -n '1,220p' "$rock_path"
      echo
    fi
  done < "$ROCKSPEC_REPORT"
  echo
  echo "Native source candidates"
  cat "$SOURCE_REPORT"
  echo
  echo "Copied snippets"
  find "$SNIPPET_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
} > "$REPORT"

IMAGE_LUA="$TORCH_DIR/share/lua/5.1/image/init.lua"
if [[ ! -f "$IMAGE_LUA" ]] && ! grep -q "function.image.scale.source" "$STAGE2D_OUT/lua_introspection.txt"; then
  echo "could not locate installed image Lua wrapper" | tee -a "$REPORT" >&2
  exit 2
fi
if [[ ! -s "$LIB_REPORT" ]]; then
  echo "could not locate installed image shared object" | tee -a "$REPORT" >&2
  exit 2
fi

echo "wrote $REPORT"
echo "wrote $SNIPPET_DIR"
