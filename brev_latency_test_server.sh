curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

echo "Making fake policy..."
uv run python scripts/create_fake_act_policy.py --output-dir outputs/fake_act_so101

echo "Starting server..."
ASYNC_SERVER_HOST=127.0.0.1 \
ASYNC_SERVER_PORT=8080 \
ASYNC_SERVER_INFERENCE_LATENCY=0 \
bash scripts/async_server.sh