#!/bin/bash
# Wrapper to start Voice Agent with proper environment
cd /Users/siddharthmishra/Downloads/nexus
export PYTHONPATH="/Users/siddharthmishra/Downloads/nexus:/Users/siddharthmishra/Downloads/nexus/services/voice-agent"
export PATH="/Users/siddharthmishra/Library/Python/3.9/bin:$PATH"

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

exec uvicorn main:app --port 8001 --app-dir services/voice-agent
