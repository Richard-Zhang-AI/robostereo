#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace/WMPO"
HOST_REPO="/nfs/rczhang/code/WMPO"
TARGET_REL="verl/workers/rollout/robwm_rollout.py"
SENTINEL=""
LOG_FILE=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/self_check_wmpo_runtime.sh [options]

Options:
  --repo PATH         Repo path inside container (default: /workspace/WMPO)
  --host-repo PATH    Host path (if visible in container) (default: /nfs/rczhang/code/WMPO)
  --target RELPATH    File to hash-check (default: verl/workers/rollout/robwm_rollout.py)
  --sentinel STR      Optional sentinel string to grep in log
  --log FILE          Optional log file used with --sentinel
  -h, --help          Show this help

Example:
  bash scripts/self_check_wmpo_runtime.sh \
    --sentinel MY_PATCH_20260213 \
    --log /workspace/WMPO/checkpoint_files/OEPL-mimicgen/coffee_wmpo_128/runs/20260213_120000/terminal.log
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="$2"; shift 2 ;;
    --host-repo)
      HOST_REPO="$2"; shift 2 ;;
    --target)
      TARGET_REL="$2"; shift 2 ;;
    --sentinel)
      SENTINEL="$2"; shift 2 ;;
    --log)
      LOG_FILE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERR] Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

ok()   { echo "[OK ] $*"; }
warn() { echo "[WARN] $*"; }
info() { echo "[INFO] $*"; }

section() {
  echo
  echo "=== $* ==="
}

must_exist() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "[ERR] Missing path: $p" >&2
    exit 1
  fi
}

section "Basic"
info "time: $(date '+%F %T %Z')"
info "host: $(hostname)"
info "user: $(whoami)"
info "pwd : $(pwd)"
info "python: $(command -v python || true)"

section "Repo Path"
must_exist "$REPO"
info "repo(raw): $REPO"
info "repo(abs): $(readlink -f "$REPO")"
if findmnt -T "$REPO" >/dev/null 2>&1; then
  info "mount:"
  findmnt -T "$REPO" | sed 's/^/  /'
else
  warn "findmnt not available or mount info not found for $REPO"
fi

section "Target File"
CONTAINER_FILE="$REPO/$TARGET_REL"
must_exist "$CONTAINER_FILE"
info "container file: $CONTAINER_FILE"
CONTAINER_SHA="$(sha256sum "$CONTAINER_FILE" | awk '{print $1}')"
info "container sha256: $CONTAINER_SHA"

HOST_FILE="$HOST_REPO/$TARGET_REL"
if [[ -f "$HOST_FILE" ]]; then
  HOST_SHA="$(sha256sum "$HOST_FILE" | awk '{print $1}')"
  info "host file     : $HOST_FILE"
  info "host sha256   : $HOST_SHA"
  if [[ "$HOST_SHA" == "$CONTAINER_SHA" ]]; then
    ok "host/container file content matches"
  else
    warn "host/container file content differs"
  fi
else
  warn "host file not visible in container: $HOST_FILE (skip host hash compare)"
fi

section "Python Import Resolution"
python - "$REPO" <<'PY'
import importlib
import inspect
import os
import sys

repo = os.path.abspath(sys.argv[1])
if repo not in sys.path:
    sys.path.insert(0, repo)

mods = [
    "verl",
    "verl.trainer.main_ppo",
    "verl.workers.rollout.robwm_rollout",
]
for m in mods:
    try:
        mod = importlib.import_module(m)
        path = inspect.getfile(mod)
        print(f"{m:40s} -> {path}")
    except Exception as e:
        print(f"{m:40s} -> [IMPORT-ERROR] {type(e).__name__}: {e}")
PY

section "Syntax Check"
python -m py_compile "$CONTAINER_FILE" \
  && ok "python syntax is valid for $TARGET_REL" \
  || warn "python syntax check failed for $TARGET_REL"

section "Running Processes"
PPO_PIDS="$(pgrep -f "python .*verl.trainer.main_ppo" || true)"
if [[ -z "${PPO_PIDS}" ]]; then
  warn "no running main_ppo process found"
else
  ok "found running main_ppo pids: $(echo "$PPO_PIDS" | tr '\n' ' ')"
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    echo "  pid=$pid"
    echo "    exe: $exe"
    echo "    cwd: $cwd"
    echo "    cmd: $cmd"
  done <<< "$PPO_PIDS"
fi

section "Sentinel Check (Optional)"
if [[ -n "$SENTINEL" ]]; then
  if [[ -z "$LOG_FILE" ]]; then
    warn "--sentinel is set but --log is empty; skip grep"
  elif [[ ! -f "$LOG_FILE" ]]; then
    warn "log file not found: $LOG_FILE"
  else
    info "grep sentinel '$SENTINEL' in $LOG_FILE"
    if grep -n "$SENTINEL" "$LOG_FILE"; then
      ok "sentinel found in log"
    else
      warn "sentinel NOT found in log"
    fi
  fi
else
  info "no sentinel provided, skip log grep"
fi

section "Quick Checklist"
echo "1) import path should point to /workspace/WMPO (not site-packages old copy)"
echo "2) hash compare should match host/container (if host path visible)"
echo "3) after code change, restart ray+trainer to load new code:"
echo "   ray stop -f"
echo "   pkill -f 'python -m verl.trainer.main_ppo' || true"
echo "4) add a unique sentinel print in code and grep log to verify runtime hit"
