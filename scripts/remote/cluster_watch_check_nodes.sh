#!/usr/bin/env bash
set -u

BASE="${HOME}/researchops/cluster_watch"
ENV_FILE="${BASE}/telegram.env"
NODES_FILE="${BASE}/nodes.tsv"
STATE_FILE="${BASE}/state.tsv"
TMP_FILE="${BASE}/state.tmp"
LOCK_FILE="${BASE}/check.lock"
LOG_FILE="${BASE}/watcher.log"

mkdir -p "${BASE}"

exec 9>"${LOCK_FILE}"
flock -n 9 || exit 0

log() {
  printf '%s %s\n' "$(date -Is)" "$*" >> "${LOG_FILE}"
}

if [[ ! -f "${ENV_FILE}" ]]; then
  log "missing ${ENV_FILE}"
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

send_tg() {
  local msg="$1"
  if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    log "telegram not configured: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing"
    return 0
  fi
  curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=${msg}" >/dev/null || log "telegram send failed"
}

discover_chat_id() {
  if [[ -n "${TELEGRAM_CHAT_ID:-}" || -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
    return 0
  fi
  local updates chat_id
  updates="$(curl -fsS "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getUpdates" 2>/dev/null)" || return 0
  chat_id="$(
    python3 -c '
import json, sys
data=json.load(sys.stdin)
for row in reversed(data.get("result", [])):
    msg=row.get("message") or row.get("channel_post") or {}
    chat=msg.get("chat") or {}
    if chat.get("id") is not None:
        print(chat["id"])
        break
' <<< "${updates}"
  )"
  if [[ -n "${chat_id}" ]]; then
    TELEGRAM_CHAT_ID="${chat_id}"
    python3 - "$ENV_FILE" "$chat_id" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
chat_id = sys.argv[2]
lines = path.read_text().splitlines()
out = []
seen = False
for line in lines:
    if line.startswith("TELEGRAM_CHAT_ID="):
        out.append(f"TELEGRAM_CHAT_ID='{chat_id}'")
        seen = True
    else:
        out.append(line)
if not seen:
    out.append(f"TELEGRAM_CHAT_ID='{chat_id}'")
path.write_text("\n".join(out) + "\n")
PY
    chmod 600 "${ENV_FILE}"
    log "telegram chat id discovered"
  else
    log "telegram chat id unavailable; send /start to the bot"
  fi
}

status_snapshot() {
  if command -v poormans >/dev/null 2>&1; then
    timeout 12s poormans 2>&1 | sed -n '1,24p'
  elif command -v research_status >/dev/null 2>&1; then
    timeout 12s research_status resources 2>&1 | sed -n '1,24p'
  else
    uptime
    free -m | sed -n '1,2p'
  fi
}

check_node() {
  local name="$1"
  local ip="$2"
  local user="$3"
  local local_hostname

  local_hostname="$(hostname)"
  if [[ "${name}" == "ofi1" && "${local_hostname}" == "uace-ofi-01" ]]; then
    echo "up|${local_hostname}|local"
    return
  fi
  if [[ "${name}" == "ofi2" && "${local_hostname}" == "uace-ofi-02" ]]; then
    echo "up|${local_hostname}|local"
    return
  fi

  local ts_ok=0
  timeout 7s tailscale ping --timeout=3s "${ip}" >/dev/null 2>&1 && ts_ok=1

  local host_out
  host_out="$(timeout 8s ssh -n -F /dev/null \
    -o BatchMode=yes \
    -o ConnectTimeout=4 \
    -o StrictHostKeyChecking=accept-new \
    "${user}@${ip}" 'hostname' 2>/dev/null)" && {
      echo "up|${host_out}|ssh+tailscale"
      return
    }

  if [[ "${ts_ok}" -eq 1 ]]; then
    echo "up|-|tailscale ping ok, ssh failed"
  else
    echo "down|-|no tailscale ping, no ssh"
  fi
}

discover_chat_id

now="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
watcher="$(hostname)"

: > "${TMP_FILE}"

while IFS=$'\t' read -r name ip user; do
  [[ -z "${name:-}" ]] && continue
  [[ "${name}" =~ ^# ]] && continue

  result="$(check_node "${name}" "${ip}" "${user}")"
  status="$(echo "${result}" | cut -d'|' -f1)"
  host="$(echo "${result}" | cut -d'|' -f2)"
  detail="$(echo "${result}" | cut -d'|' -f3-)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${name}" "${ip}" "${status}" "${host}" "${detail}" "${now}" >> "${TMP_FILE}"
done < "${NODES_FILE}"

if [[ ! -f "${STATE_FILE}" ]]; then
  cp "${TMP_FILE}" "${STATE_FILE}"
  msg="Cluster watcher initialized on ${watcher}
${now}

Current:
$(awk -F'\t' '{printf "%s %s %s %s\n", $1, $3, $4, $5}' "${STATE_FILE}")

Snapshot:
$(status_snapshot)"
  send_tg "${msg}"
  exit 0
fi

changes=""
while IFS=$'\t' read -r name ip status host detail ts; do
  prev_line="$(awk -F'\t' -v n="${name}" '$1==n {print; exit}' "${STATE_FILE}")"
  prev_status="$(echo "${prev_line}" | awk -F'\t' '{print $3}')"

  if [[ -z "${prev_status}" ]]; then
    changes+="${name}: NEW -> ${status} (${detail})"$'\n'
  elif [[ "${prev_status}" != "${status}" ]]; then
    changes+="${name}: ${prev_status} -> ${status} (${detail})"$'\n'
  fi
done < "${TMP_FILE}"

cp "${TMP_FILE}" "${STATE_FILE}"

if [[ -n "${changes}" ]]; then
  msg="Cluster node status changed on ${watcher}
${now}

${changes}
Current:
$(awk -F'\t' '{printf "%s %s %s %s\n", $1, $3, $4, $5}' "${STATE_FILE}")

Snapshot:
$(status_snapshot)"
  send_tg "${msg}"
fi
