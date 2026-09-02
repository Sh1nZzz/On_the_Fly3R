#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

prepare_repository() {
    local target_dir="$1"
    local repo_url="$2"

    if [ -d "${target_dir}/.git" ]; then
        echo "${target_dir} already exists."
    elif [ -e "${target_dir}" ]; then
        echo "Error: ${target_dir} exists but is not a Git checkout." >&2
        return 1
    else
        echo "Cloning ${repo_url} into ${target_dir}..."
        git clone --recursive "${repo_url}" "${target_dir}"
    fi

    if [ -f "${target_dir}/.gitmodules" ]; then
        echo "Initializing submodules in ${target_dir}..."
        git -C "${target_dir}" submodule sync --recursive
        git -C "${target_dir}" submodule update --init --recursive
    fi
}

apply_repository_patch() {
    local target_dir="$1"
    local patch_file="$2"

    if git -C "${target_dir}" apply --check "${patch_file}"; then
        git -C "${target_dir}" apply "${patch_file}"
        echo "Applied compatibility patch $(basename "${patch_file}") to ${target_dir}."
    elif git -C "${target_dir}" apply --reverse --check "${patch_file}"; then
        echo "Compatibility patch $(basename "${patch_file}") is already applied to ${target_dir}."
    else
        echo "Error: $(basename "${patch_file}") is incompatible with the checked-out ${target_dir} version." >&2
        return 1
    fi
}

echo "Installing base requirements..."
python -m pip install -r requirements.txt

echo "Preparing local external repositories ignored by Git..."
prepare_repository "Pi3" "https://github.com/yyfz/Pi3.git"
prepare_repository "SupScene" "https://github.com/Suxilan/SupScene.git"
apply_repository_patch "SupScene" "${PROJECT_ROOT}/patches/supscene_absolute_dinov2_path.patch"

if [ ! -f "SupScene/third_party/dinov2/hubconf.py" ]; then
    echo "Error: SupScene's DINOv2 submodule is incomplete: SupScene/third_party/dinov2/hubconf.py is missing." >&2
    exit 1
fi

if [ -d "Pi3" ] && { [ -f "Pi3/setup.py" ] || [ -f "Pi3/pyproject.toml" ]; }; then
    echo "Installing Pi3 in editable mode..."
    python -m pip install -e ./Pi3
fi

if [ "${ON_THE_FLY3R_INSTALL_MAPANYTHING:-0}" = "1" ]; then
    prepare_repository "map-anything" "https://github.com/facebookresearch/map-anything.git"
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
