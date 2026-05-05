echo "Making fake policy..."
uv run python scripts/create_fake_act_policy.py --output-dir outputs/fake_act_so101

echo "Starting server..."
ASYNC_SERVER_HOST=127.0.0.1 \
ASYNC_SERVER_PORT=8080 \
ASYNC_SERVER_INFERENCE_LATENCY=0 \
bash scripts/async_server.sh