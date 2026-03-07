# Intrinsic Dimension Reimplementation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

PyTorch reimplementation of the ICLR 2018 paper *Measuring the Intrinsic Dimension of Objective Landscapes*, with classic MNIST/CIFAR/RL experiments and an additional ViT + LoRA + intrinsic-dimension extension for transfer learning studies.

- Reimplements random-subspace training via `theta = theta_0 + P @ d`
- Measures `d_int90`, the smallest subspace dimension that reaches 90% of baseline performance
- Includes dense, sparse, and Fastfood projection backends
- Covers classic paper experiments plus optional ViT transfer experiments

![MNIST FC intrinsic dimension](results/mnist_fc_intrinsic_dim.png)

## Overview

The core idea of intrinsic-dimension training is simple:

```text
full parameter vector theta in R^D
    ->
train only a low-dimensional vector d in R^small
    ->
recover theta with a random projection
    ->
theta = theta_0 + P @ d
```

Instead of optimizing all model parameters directly, the model is trained inside a random low-dimensional subspace. This lets us ask how many degrees of freedom are actually needed before the model reaches near-baseline performance.

The main metric used throughout the repository is:

- `d_int90`: the smallest tested dimension that reaches 90% of the baseline score

## Repository Scope

This repository has two layers:

1. The main paper reimplementation:
   FC, LeNet, CIFAR, RL, regularization, projection comparisons, and the original intrinsic-dimension measurement workflow.
2. An advanced extension:
   ViT transfer experiments that compare full fine-tuning, linear probing, LoRA, module-matched intrinsic dimension, and full-parameter intrinsic dimension.

If you only care about the original paper, you can ignore the ViT-specific scripts. If you want the newer transfer-learning extension, see `experiments/vit_intrinsic_lora.py`, `scripts/run_vit_intrinsic_plan.py`, and `docs/vit_id_lora_execution_plan.md`.

## Key Results From the Paper Reimplementation

| Problem | Model | Parameters | d_int90 | Compression |
| --- | --- | ---: | ---: | ---: |
| Inverted Pendulum | FC | 562 | 4 | 140x |
| CartPole | FC | 199,210 | 25 | 7,968x |
| MNIST | LeNet | 44,426 | 275 | 161x |
| MNIST | FC | 199,210 | 750 | 266x |
| CIFAR-10 | LeNet | 62,006 | 2,900 | 21x |
| Atari Pong | ConvNet | 1,005,974 | 6,000 | 168x |
| CIFAR-10 | FC | 1,055,610 | 9,000 | 117x |
| ImageNet | SqueezeNet | 1,248,424 | >500,000 | <2.5x |

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Compatibility notes:

- `requirements.txt` is the main environment for current use
- `requirements.vit_py37.txt` is kept for older Python 3.7 / older PyTorch environments
- Most MNIST, CIFAR-10, CIFAR-100, and Flowers-102 loaders can download datasets automatically
- ImageNet experiments require you to prepare the dataset manually

## Quick Start

Fastest sanity check:

```bash
python test_compatibility.py
python experiments/toy_problem.py
```

Common paper-reproduction runs:

```bash
# MNIST fully-connected network
python experiments/mnist_fc.py

# MNIST LeNet
python experiments/mnist_lenet.py

# FC architecture sweep (paper Figure 3)
python experiments/mnist_fc_variants.py

# Direct small network vs subspace training (paper Figure 4)
python experiments/direct_vs_subspace.py

# Projection method comparison
python experiments/projection_comparison.py

# Run several core experiments together
python scripts/run_all_experiments.py --experiments toy,mnist_fc,mnist_lenet --quick
```

Generate summary figures after experiments:

```bash
python scripts/generate_figures.py
python scripts/generate_paper_figures.py
```

## Main Experiments

### Toy Problem

Shows that the optimization succeeds once the subspace dimension reaches the true intrinsic dimension.

```bash
python experiments/toy_problem.py
```

### MNIST

```bash
python experiments/mnist_fc.py
python experiments/mnist_lenet.py
python experiments/mnist_shuffled_pixel.py
python experiments/mnist_shuffled_label.py
```

These runs reproduce the core story of the paper:

- FC and LeNet have very different parameter counts but much closer intrinsic dimensions than naive size arguments suggest
- destroying spatial structure removes LeNet's advantage
- random labels push the required intrinsic dimension dramatically upward

### CIFAR-10

```bash
python experiments/cifar_resnet.py --model lenet
python experiments/cifar_resnet.py --model resnet
```

### Reinforcement Learning

```bash
python experiments/rl_experiments.py --env cartpole
python experiments/rl_experiments.py --env pendulum
```

### ImageNet SqueezeNet

```bash
python experiments/imagenet_squeezenet.py --data_dir /path/to/imagenet
```

This is the most computationally demanding classic-paper experiment.

## Advanced ViT + LoRA + Intrinsic-Dimension Extension

This repository also includes a modern extension that compares:

- full fine-tuning
- linear probing
- LoRA
- module-matched intrinsic-dimension training
- full-parameter intrinsic-dimension training

Main entry points:

- `experiments/vit_intrinsic_lora.py`
- `scripts/run_vit_intrinsic_plan.py`
- `scripts/tune_vit_hparams.py`
- `scripts/run_vit_all_in_one.py`
- `docs/vit_id_lora_execution_plan.md`

Typical commands:

```bash
# Print the ViT experiment plan without executing it
python scripts/run_vit_intrinsic_plan.py --phase all --dry_run

# Smoke test the ViT pipeline
python scripts/test_vit_intrinsic_pipeline.py --epochs 1 --num_batches 2

# One-click full plan preview
python scripts/run_vit_all_in_one.py --dry_run
```

## Project Structure

```text
intrinsic_dim/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements.vit_py37.txt
├── test_compatibility.py
├── docs/
│   └── vit_id_lora_execution_plan.md
├── experiments/
│   ├── toy_problem.py
│   ├── mnist_fc.py
│   ├── mnist_fc_variants.py
│   ├── mnist_lenet.py
│   ├── mnist_shuffled_pixel.py
│   ├── mnist_shuffled_label.py
│   ├── direct_vs_subspace.py
│   ├── projection_comparison.py
│   ├── cifar_resnet.py
│   ├── rl_experiments.py
│   ├── imagenet_squeezenet.py
│   ├── ablation_regularization.py
│   ├── ablation_convnet_variants.py
│   └── vit_intrinsic_lora.py
├── scripts/
│   ├── run_all_experiments.py
│   ├── generate_figures.py
│   ├── generate_paper_figures.py
│   ├── run_vit_intrinsic_plan.py
│   ├── tune_vit_hparams.py
│   ├── test_vit_intrinsic_pipeline.py
│   └── run_vit_all_in_one.py
├── src/
│   ├── projections/
│   ├── models/
│   └── utils/
├── notebooks/
└── results/
```

## Architecture And Implementation Highlights

### 1. `SubspaceModel`

`src/models/subspace.py` is the core wrapper. It:

- freezes the base model parameters at `theta_0`
- allocates a trainable low-dimensional vector `theta`
- projects `theta` back into the selected parameter space
- reconstructs the effective model weights on each forward pass

Important implementation detail:

- On PyTorch 2.0+, it uses `torch.func.functional_call`
- On older PyTorch versions, it falls back to a parameter-swap path so gradients still flow correctly

That backward-compatible design is one of the main engineering points of this repository.

### 2. Projection Backends

The repository implements three projection families:

- `DenseProjection`: straightforward random Gaussian projection
- `SparseProjection`: memory-friendlier sparse random projection
- `FastfoodProjection`: structured transform with `O(D log D)` compute and `O(D)` memory

For large models, Fastfood is the practical choice.

### 3. Metrics Layer

`src/utils/metrics.py` contains the logic for:

- finding `d_int90`
- interpolating threshold crossings
- estimating compression ratio
- bootstrap-based uncertainty estimation across seeds

### 4. Training Utilities

`src/utils/training.py` provides reusable loops for training and evaluation. Experiment scripts build their dimension sweeps on top of these utilities and cache outputs into `results/`.

## Data, Outputs, And Caching

What is intentionally not tracked in Git:

- downloaded datasets under `data/`
- temporary checkpoints
- large auto-run result directories
- notebook checkpoints and Python cache files

What is kept in the repository:

- code
- documentation
- selected result figures and JSON summaries useful for inspection

Most experiment scripts cache outputs under `results/` so repeated runs can skip recomputation unless you pass `--force`.

## Citation

If you use this repository, please cite the original paper:

```bibtex
@inproceedings{li2018measuring,
  title={Measuring the Intrinsic Dimension of Objective Landscapes},
  author={Li, Chunyuan and Farkhoor, Heerad and Liu, Rosanne and Yosinski, Jason},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
