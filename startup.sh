#!/bin/bash
trap 'kill 0' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SCRIPT_DIR/src"

inference_args=()
if [ -n "${INFERENCE_LLM_PROVIDER:-}" ]; then
    inference_args+=(--provider "$INFERENCE_LLM_PROVIDER")
fi
if [ -n "${INFERENCE_LLM_MODEL:-}" ]; then
    inference_args+=(--model "$INFERENCE_LLM_MODEL")
fi

python -m coda.inference.agent "${inference_args[@]}" &

echo "Waiting for inference agent..."
until curl -sf http://localhost:5123/health > /dev/null 2>&1; do
    sleep 1
done
echo "Inference agent ready."

python -m coda.app &

echo "Waiting for web application..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    sleep 1
done
echo "CODA is running at http://localhost:8000"

wait
