curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"