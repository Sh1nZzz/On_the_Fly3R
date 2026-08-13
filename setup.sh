#!/usr/bin/env bash
set -euo pipefail

clone_if_missing() {
    local target_dir="$1"
    local repo_url="$2"

    if [ -d "${target_dir}/.git" ]; then
        echo "${target_dir} already exists; skipping clone."
        return
    fi

    if [ -e "${target_dir}" ]; then
        echo "${target_dir} exists but is not a Git checkout; leaving it untouched."
        return
    fi

    echo "Cloning ${repo_url} into ${target_dir}..."
    git clone "${repo_url}" "${target_dir}"
}

echo "Installing base requirements..."
python -m pip install -r requirements.txt

echo "Preparing local external repositories ignored by Git..."
clone_if_missing "Pi3" "https://github.com/yyfz/Pi3.git"
clone_if_missing "SupScene" "https://github.com/Suxilan/SupScene.git"

if [ -d "Pi3" ] && { [ -f "Pi3/setup.py" ] || [ -f "Pi3/pyproject.toml" ]; }; then
    echo "Installing Pi3 in editable mode..."
    python -m pip install -e ./Pi3
fi

if [ "${ON_THE_FLY3R_INSTALL_MAPANYTHING:-0}" = "1" ]; then
    clone_if_missing "map-anything" "https://github.com/facebookresearch/map-anything.git"
    if [ -d "map-anything" ]; then
        python -m pip install -e ./map-anything
    fi
fi

echo "Installing on-the-fly3r in editable mode..."
python -m pip install -e .

cat <<'MSG'

Installation complete.

Remember to provide model weights locally:
  - Pi3/Pi3x: --model_checkpoint /path/to/model.safetensors or ON_THE_FLY3R_PI3_CHECKPOINT
  - SupScene: --supscene_weights /path/to/dinov2_scpp_supscene_1536.pth
External repositories and weights are ignored by Git.
MSG
