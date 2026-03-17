#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# NEXUS — Stop All Services
# Stops all agent services and optionally infrastructure.
# Usage: bash scripts/stop_all.sh           # stop agents only
#        bash scripts/stop_all.sh --all     # stop agents + docker
# ══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Colours ──
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_DIM="\033[2m"
C_RED="\033[91m"
C_GREEN="\033[92m"
C_YELLOW="\033[93m"
C_CYAN="\033[96m"

PID_DIR="$PROJECT_ROOT/.pids"

STOP_DOCKER=false
if [[ "${1:-}" == "--all" ]]; then
    STOP_DOCKER=true
fi

echo -e "${C_BOLD}${C_CYAN}NEXUS — Stopping Services${C_RESET}"
echo -e "${C_DIM}────────────────────────────────────────${C_RESET}"

# ══════════════════════════════════════════════════════════════
# Stop Agent Services (no associative arrays — bash 3.x compat)
# ══════════════════════════════════════════════════════════════

stop_service() {
    local name=$1
    local port=$2

    # Try PID file first
    pid=""
    pid_file="$PID_DIR/${name}.pid"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file" 2>/dev/null || true)
    fi

    # Also find by port
    port_pids=$(lsof -ti ":${port}" -sTCP:LISTEN 2>/dev/null || true)

    # Combine
    all_pids="$pid $port_pids"
    all_pids=$(echo "$all_pids" | tr ' ' '\n' | sort -u | grep -v '^$' || true)

    if [ -n "$all_pids" ]; then
        for p in $all_pids; do
            if kill -0 "$p" 2>/dev/null; then
                kill "$p" 2>/dev/null || true
                echo -e "  ${C_GREEN}[STOPPED]${C_RESET} ${name} (PID ${p}, port ${port})"
            fi
        done
    else
        echo -e "  ${C_DIM}[SKIP]${C_RESET}    ${name} (not running)"
    fi

    # Clean up PID file
    rm -f "$pid_file" 2>/dev/null || true
}

stop_service "voice-agent"    8001
stop_service "language-agent"  8002
stop_service "fusion-agent"    8007
stop_service "api-gateway"     8000

# Also kill any stray uvicorn processes for this project
stray_pids=$(pgrep -f "uvicorn.*services/" 2>/dev/null || true)
if [ -n "$stray_pids" ]; then
    echo -e "\n  ${C_YELLOW}Cleaning up stray uvicorn processes...${C_RESET}"
    for p in $stray_pids; do
        if kill -0 "$p" 2>/dev/null; then
            kill "$p" 2>/dev/null || true
            echo -e "  ${C_DIM}  Killed PID ${p}${C_RESET}"
        fi
    done
fi

# ══════════════════════════════════════════════════════════════
# Stop Infrastructure (Docker)
# ══════════════════════════════════════════════════════════════

if [ "$STOP_DOCKER" = true ]; then
    echo -e "\n${C_BOLD}Stopping Docker infrastructure...${C_RESET}"
    if command -v docker &> /dev/null; then
        docker compose down 2>&1 | while read -r line; do echo "  $line"; done
        echo -e "  ${C_GREEN}Docker services stopped.${C_RESET}"
    else
        echo -e "  ${C_DIM}Docker not found — skipping.${C_RESET}"
    fi
else
    echo -e "\n${C_DIM}Docker infrastructure left running. Use --all to stop everything.${C_RESET}"
fi

echo -e "\n${C_GREEN}${C_BOLD}All services stopped.${C_RESET}\n"
