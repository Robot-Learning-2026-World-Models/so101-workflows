curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

FAKE_POLICY_DIR="${FAKE_POLICY_DIR:-outputs/fake_act_so101}"

if [[ -f "${FAKE_POLICY_DIR}/config.json" && -f "${FAKE_POLICY_DIR}/model.safetensors" ]]; then
  echo "Using existing fake policy at ${FAKE_POLICY_DIR}"
else
  echo "Making fake policy at ${FAKE_POLICY_DIR}..."
  uv run python scripts/create_fake_act_policy.py --output-dir "${FAKE_POLICY_DIR}"
fi

echo "Starting server..."
ASYNC_SERVER_HOST=127.0.0.1 \
ASYNC_SERVER_PORT=8080 \
ASYNC_SERVER_INFERENCE_LATENCY=0 \
bash scripts/async_server.sh
