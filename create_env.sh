ENV_NAME="${ENV_NAME:-1}"

micromamba create -y -n "${ENV_NAME}" python=3.11
micromamba activate "${ENV_NAME}"
pip install -r requirements.txt