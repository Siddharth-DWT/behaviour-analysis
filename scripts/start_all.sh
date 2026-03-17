#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# NEXUS — Start All Services
# Starts infrastructure (Docker) then all 4 agent services.
# Usage: bash scripts/start_all.sh
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
C_BLUE="\033[94m"
C_CYAN="\033[96m"

# ── Load .env if present ──
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${C_DIM}Loading environment from .env${C_RESET}"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

PID_DIR="$PROJECT_ROOT/.pids"
LOG_DIR="$PROJECT_ROOT/.logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

UVICORN="$(which uvicorn 2>/dev/null || echo "")"
if [ -z "$UVICORN" ]; then
    echo -e "${C_RED}uvicorn not found. Install: pip3 install uvicorn${C_RESET}"
    exit 1
fi

echo -e "${C_BOLD}${C_CYAN}NEXUS — Starting All Services${C_RESET}"
echo -e "${C_DIM}────────────────────────────────────────${C_RESET}"

# ══════════════════════════════════════════════════════════════
# Step 1: Infrastructure (Docker Compose)
# ══════════════════════════════════════════════════════════════
echo -e "\n${C_BOLD}[1/5] Starting infrastructure (Docker Compose)...${C_RESET}"

if ! command -v docker &> /dev/null; then
    echo -e "${C_YELLOW}  Docker not found — skipping infrastructure.${C_RESET}"
    echo -e "${C_YELLOW}  Agents will run without PostgreSQL/Redis.${C_RESET}"
else
    docker compose up -d 2>&1 | while read -r line; do echo "  $line"; done

    # Wait for PostgreSQL to be ready
    echo -ne "  Waiting for PostgreSQL..."
    for i in $(seq 1 30); do
        if docker compose exec -T postgres pg_isready -q 2>/dev/null; then
            echo -e " ${C_GREEN}ready${C_RESET}"
            break
        fi
        echo -n "."
        sleep 1
        if [ "$i" -eq 30 ]; then
            echo -e " ${C_YELLOW}timeout (may still be starting)${C_RESET}"
        fi
    done

    # Wait for Redis/Valkey
    echo -ne "  Waiting for Valkey (Redis)..."
    for i in $(seq 1 15); do
        if docker compose exec -T valkey redis-cli ping 2>/dev/null | grep -q PONG; then
            echo -e " ${C_GREEN}ready${C_RESET}"
            break
        fi
        echo -n "."
        sleep 1
        if [ "$i" -eq 15 ]; then
            echo -e " ${C_YELLOW}timeout${C_RESET}"
        fi
    done
fi

# ══════════════════════════════════════════════════════════════
# Step 2-5: Start Agent Services
# ══════════════════════════════════════════════════════════════

start_service() {
    local name=$1
    local port=$2
    local app_dir=$3
    local step=$4

    echo -e "\n${C_BOLD}[${step}/5] Starting ${name} on port ${port}...${C_RESET}"

    # Check if port is already in use
    if lsof -i ":${port}" -sTCP:LISTEN &>/dev/null; then
        local existing_pid
        existing_pid=$(lsof -ti ":${port}" -sTCP:LISTEN 2>/dev/null | head -1)
        echo -e "  ${C_YELLOW}Port ${port} already in use (PID ${existing_pid}). Skipping.${C_RESET}"
        echo "$existing_pid" > "$PID_DIR/${name}.pid"
        return 0
    fi

    # Start uvicorn in background
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/$app_dir" \
        nohup "$UVICORN" main:app \
        --port "$port" \
        --app-dir "$app_dir" \
        > "$LOG_DIR/${name}.log" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_DIR/${name}.pid"
    echo -e "  ${C_DIM}PID: ${pid}, Log: .logs/${name}.log${C_RESET}"
}

start_service "voice-agent"    8001 "services/voice-agent"    2
start_service "language-agent"  8002 "services/language-agent"  3
start_service "fusion-agent"    8007 "services/fusion-agent"    4
start_service "api-gateway"     8000 "services/api-gateway"     5

# ══════════════════════════════════════════════════════════════
# Health Check Loop
# ══════════════════════════════════════════════════════════════
echo -e "\n${C_BOLD}Waiting for services to be ready...${C_RESET}"

MAX_WAIT=120  # seconds
INTERVAL=3
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    ALL_READY=true
    STATUS_LINE=""

    for pair in "voice-agent:8001" "language-agent:8002" "fusion-agent:8007" "api-gateway:8000"; do
        name="${pair%%:*}"
        port="${pair##*:}"
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            STATUS_LINE+="  ${C_GREEN}[OK]${C_RESET} ${name}"
        else
            STATUS_LINE+="  ${C_YELLOW}[..]${C_RESET} ${name}"
            ALL_READY=false
        fi
    done

    # Print status
    echo -ne "\r\033[K  ${ELAPSED}s:${STATUS_LINE}  "

    if [ "$ALL_READY" = true ]; then
        echo ""
        echo -e "\n${C_GREEN}${C_BOLD}All services are ready!${C_RESET}"
        echo -e "${C_DIM}────────────────────────────────────────${C_RESET}"
        echo -e "  Voice Agent:    http://localhost:8001"
        echo -e "  Language Agent: http://localhost:8002"
        echo -e "  Fusion Agent:   http://localhost:8007"
        echo -e "  API Gateway:    http://localhost:8000"
        echo -e "  Dashboard:      http://localhost:3000 (start separately: cd dashboard && npx vite)"
        echo -e "${C_DIM}────────────────────────────────────────${C_RESET}"
        echo -e "  Logs:   ${C_DIM}.logs/<service>.log${C_RESET}"
        echo -e "  PIDs:   ${C_DIM}.pids/<service>.pid${C_RESET}"
        echo -e "  Stop:   ${C_DIM}bash scripts/stop_all.sh${C_RESET}"
        echo -e "  Test:   ${C_DIM}python3 scripts/test_pipeline.py${C_RESET}"
        exit 0
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo -e "\n${C_YELLOW}${C_BOLD}Timeout after ${MAX_WAIT}s. Some services may still be starting.${C_RESET}"
echo -e "Check logs in .logs/ for errors."

# Print final status
echo -e "\nFinal status:"
for pair in "voice-agent:8001" "language-agent:8002" "fusion-agent:8007" "api-gateway:8000"; do
    name="${pair%%:*}"
    port="${pair##*:}"
    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo -e "  ${C_GREEN}[OK]${C_RESET}   ${name} (:${port})"
    else
        echo -e "  ${C_RED}[FAIL]${C_RESET} ${name} (:${port}) — check .logs/${name}.log"
    fi
done

exit 1
