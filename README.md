# On-the-Fly3R

On-the-Fly3R is an incremental 3D reconstruction framework built around 3D
vision foundation models. It reconstructs an initial image window, retrieves
relevant reference frames for new images, aligns every local reconstruction to
the global map, and incrementally fuses the result.

The current release supports:

- Pi3, Pi3x, VGGT, VGGT-Omega, and MapAnything inference adapters;
- SupScene-based image retrieval and retrieval-guided dynamic batching;
- robust point-based Sim(3) alignment, validation, and reference-pruning retry;
- optional final SE(3) pose-graph optimization with GTSAM;
- camera-pose, run-summary, and PLY point-cloud export;
- an online Viser viewer.

## Installation

The recommended environment is Linux, Python 3.11, PyTorch, and a CUDA-capable
GPU. Install PyTorch and TorchVision for your CUDA version first, then run:

```bash
git clone https://github.com/Sh1nZzz/On-the-Fly3R.git
cd On-the-Fly3R

conda create -n on_the_fly3r python=3.11 -y
conda activate on_the_fly3r

# Install PyTorch and TorchVision for your CUDA version before this step.
bash setup.sh
```

`setup.sh` installs this project and prepares the external `Pi3/` and
`SupScene/` repositories. Model checkpoints and datasets are not included in
this repository.

Provide the local checkpoints explicitly:

- Pi3/Pi3x: `--model_checkpoint /path/to/model.safetensors`
- SupScene: `--supscene_weights /path/to/dinov2_scpp_supscene_1536.pth`

## Quick start

Place an ordered image sequence in one directory and run:

```bash
python run_on_the_fly3r.py \
  --image_dir /path/to/images \
  --output_dir outputs/example \
  --model Pi3x \
  --model_checkpoint /path/to/model.safetensors \
  --supscene_weights /path/to/dinov2_scpp_supscene_1536.pth \
  --inference_device cuda:0 \
  --retrieval_device cuda:0 \
  --enable_register_ref_pruning_retry \
  --enable_pose_graph_optimization \
  --save_ply
```

Incoming images are grouped with retrieval-guided dynamic batching. Alignment
uses compact point correspondences and robust Sim(3) estimation. When enabled,
pose-graph optimization runs once after incremental reconstruction finishes.

## Configuration file

For repeatable experiments, edit the `editable` section in
`configs/incremental_ablation_template.yaml`, then select a case:

```bash
python run_on_the_fly3r.py \
  --config configs/incremental_ablation_template.yaml \
  --case ab05_all_on
```

Command-line arguments override values loaded from the YAML file.

## Online viewer

Use the same reconstruction arguments with the Viser entrypoint:

```bash
python view_on_the_fly3r.py \
  --config configs/incremental_ablation_template.yaml \
  --case ab05_all_on \
  --port 8080
```

Open `http://localhost:8080` in a browser after the server starts.

## Outputs

Each run writes its artifacts under `--output_dir`. Depending on the enabled
options, these include:

- `camera_poses.npz`: reconstructed camera poses;
- `summary.json`: resolved settings, timing, and reconstruction statistics;
- `resolved_config.json`: the effective configuration for the run;
- `batch_logs.json`: optional per-batch diagnostics;
- `global_points.ply`: optional fused point cloud;
- `pose_graph_debug/`: optional pose-graph diagnostics.

Evaluate exported poses against COLMAP ground truth with:

```bash
python evaluate_poses.py \
  --pred_npz outputs/example/camera_poses.npz \
  --gt_colmap /path/to/images.txt
```

## Python API

```python
from on_the_fly3r import IncrementalReconstructor, ReconstructionConfig
```

`ReconstructionConfig` defines the supported runtime configuration, while
`IncrementalReconstructor` exposes bootstrap, incremental batch processing,
final pose-graph optimization, and export operations.

## Repository layout

```text
on_the_fly3r/       reconstruction pipeline and runtime components
third_party_codes/ minimal vendored runtime utilities
evaluation/         pose evaluation
configs/            reproducible run configurations
tests/              regression tests
```

External repositories, weights, datasets, caches, and generated outputs should
remain outside Git.

## Development checks

```bash
python -m pip install pytest
git diff --check
python -m compileall -q on_the_fly3r evaluation third_party_codes
python -m pytest -q tests/test_refactor_baseline.py
```
