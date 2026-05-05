On laptop:
brev ssh-key and add it to GitHub → Settings → SSH and GPG keys

On brev:
Clone using ssh: `git@github.com:Robot-Learning-2026-World-Models/so101-workflows.git`
`./brev_setup.sh`
`./brev_latency_test.sh`

On laptop:
`brev port-forward <instance-name> --port 18080:8080`

`SERVER_ADDRESS=127.0.0.1:18080 \
PRETRAINED_NAME_OR_PATH=outputs/fake_act_so101 \
POLICY_TYPE=act \
POLICY_DEVICE=cuda \
CLIENT_DEVICE=cpu \
DURATION_S=30 \
CAMERA_WIDTH=160 \
CAMERA_HEIGHT=120 \
VERBOSE=false \
ROBOT_LOG_EVERY=0 \
bash scripts/fake_async_latency_client.sh`