# BioKinema: Physically Grounded Generative Modeling of All-Atom Biomolecular Dynamics

[![Paper](https://img.shields.io/badge/Paper-bioRxiv-green)](https://www.biorxiv.org/content/10.64898/2026.02.15.705956v1)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green?style=flat-square)](./LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red?style=flat-square)](./DATA_LICENSE)
[![GitHub Link](https://img.shields.io/badge/GitHub-blue?style=flat-square&logo=github)](https://github.com/IDEA-XL/BioKinema)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Weights-yellow?style=flat-square)](https://huggingface.co/fengb/BioKinema)

## Introduction

**BioKinema** is a physically grounded generative model that predicts continuous-time, all-atom biomolecular trajectories at a fraction of the cost of traditional molecular dynamics (MD) simulations.

Predicting the kinetic pathways of biomolecular systems at all-atom resolution is crucial for understanding protein function and drug efficacy, yet this task is hindered by the immense computational cost of conventional MD simulations. While deep learning has revolutionized static structure prediction and equilibrium ensemble sampling, simulating the kinetics of conformational transitions remains a critical challenge.

BioKinema addresses these challenges through two key innovations. First, it utilizes a **physically grounded diffusion architecture** with temporal attention mechanisms derived from Langevin dynamics. The temporal-attention bias follows a stretched-exponential decay `B_ij = -λ |t_i - t_j|^β`, where `λ` is a per-head learnable decay and `β` is a fixed time-scaling exponent selected per model variant. Second, it employs a **hierarchical forecasting-and-interpolation strategy** to overcome the error accumulation that often plagues long-horizon generation.

Through extensive validation, we demonstrate that BioKinema generates physically stable and dynamically accurate trajectories suitable for rigorous downstream analysis. The model captures key conformational transitions related to protein function. For protein-ligand complex systems, it successfully elucidates mechanisms such as ligand-driven conformational changes and allosteric interactions. Furthermore, BioKinema leverages enhanced sampling data to predict rare kinetic events, emerging as a powerful tool for estimating ligand unbinding pathways.

## Installation

```bash
# Clone the repository
git clone https://github.com/IDEA-XL/BioKinema.git
cd BioKinema

# Install dependencies
conda env create -f environment.yml
conda activate biokinema
```

### Optional Dependencies

For optimal performance, you may also configure the following environment variables in **inference.sh**:

```bash
export CUTLASS_PATH=/path/to/cutlass
export CUDA_HOME=/path/to/cuda
```

### Download Checkpoints

Trained checkpoints are available at https://huggingface.co/fengb/BioKinema. Place them under `./checkpoints/`:

| Checkpoint | For | `β` |
|------------|-----|-----|
| `BioKinema_atlas+misato+mdposit_sqrt.pt` | protein–ligand complexes and **short-time** MD | 0.5 |
| `BioKinema_CATH+octapeptide_beta0.25.pt` | **long-time, single-chain** protein MD | 0.25 |

The exponent `β` **must match the checkpoint** at inference time (pass it via `--beta`).

## Usage

### Basic Inference

Run trajectory generation using the inference script:

```bash
bash inference.sh \
    --checkpoint_path ./checkpoints/BioKinema_atlas+misato+mdposit_sqrt.pt \
    --dump_dir ./output \
    --input_file ./experiments/atlas_benchmark/init_frames/7lp1_A_R1_0.cif \
    --beta 0.5
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--checkpoint_path` | Path to the trained model checkpoint |
| `--dump_dir` | Directory for saving output trajectories |
| `--input_file` | Path to input **.pdb** or **.cif** file containing initial structure |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--beta` | 0.5 | Temporal-attention exponent; **must match the checkpoint** (0.5 for `sqrt`, 0.25 for `beta=0.25`) |
| `--N_sample` | 1 | Number of trajectory samples to generate |
| `--coarse_frame_num` | 50 | Number of coarse frames including the initial structure |
| `--coarse_interval` | 2 | Temporal spacing between coarse frames (in nanosecond) |
| `--fine_frame_num` | 1 | Number of sub-intervals per coarse interval (1 means no interpolation) |
| `--W_H` | 1 | History window size for coarse forecasting |
| `--W_G` | 50 | Generation window size for coarse forecasting |
| `--N_step` | 20 | Number of diffusion steps |
| `--N_cycle` | 10 | Number of model cycles |
| `--seed` | 101 | Random seed |
| `--lambda` | 1.75 | Noise scale parameter |
| `--eta` | 1.5 | Step scale parameter |

### Example Commands

**Standard trajectory generation:**

```bash
bash inference.sh \
    --checkpoint_path ./checkpoints/BioKinema_atlas+misato+mdposit_sqrt.pt \
    --dump_dir ./results/trajectory \
    --input_file ./experiments/atlas_benchmark/init_frames/7lp1_A_R1_0.cif \
    --beta 0.5 \
    --coarse_frame_num 50 \
    --coarse_interval 1 \
    --fine_frame_num 1
```

**Generation with interpolation:**

```bash
bash inference.sh \
    --checkpoint_path ./checkpoints/BioKinema_atlas+misato+mdposit_sqrt.pt \
    --dump_dir ./results/highres \
    --input_file ./experiments/atlas_benchmark/init_frames/7lp1_A_R1_0.cif \
    --beta 0.5 \
    --coarse_frame_num 50 \
    --coarse_interval 10 \
    --fine_frame_num 10
```

### Hierarchical Generation Pipeline

BioKinema employs a two-stage hierarchical generation approach.

**Stage 1: Coarse-grained Forecasting** generates the coarse-grained trajectory at coarse temporal resolution. The model autoregressively predicts future frames using a sliding window approach, where `W_H` controls the history context and `W_G` determines the batch size for each generation step.

**Stage 2: Fine-grained Interpolation** refines the trajectory by filling in intermediate frames between coarse keyframes. This stage is activated when `fine_frame_num > 1`, producing smoother and more temporally detailed trajectories.

## Training

BioKinema is trained with a single **trajectory-generation** procedure (`train_trajectory.sh`) that produces both released checkpoints by varying the data and the exponent `β`:

| Checkpoint | Data | `β` | Dynamics losses |
|------------|------|-----|-----------------|
| `sqrt` | Atlas + MISATO + MDposit | 0.5 | RMSF / RelRMSF / LocalRMSF / ACF / ensemble |
| `beta=0.25` | MSR (CATH / octapeptides) | 0.25 | + TICA-dynamics |

Every training config reads precomputed Pairformer embeddings (regenerate with `scripts/encode_embeddings.sh`); data roots are set via `BIOKINEMA_*` environment variables. See **[docs/data_and_training.md](docs/data_and_training.md)** for the full download → preprocess → embeddings → train walkthrough.

## Data

Data sources are released differently depending on preprocessing complexity:

- **Atlas & MSR** — preprocessing scripts are published directly (`scripts/data_prep/`); the MSR datasets additionally build MSM caches (`scripts/msm/`) for the TICA-dynamics loss. You download the raw MD and run the scripts. See **[docs/data_msm.md](docs/data_msm.md)**.
- **MISATO / MDposit / unbinding** — released as a compressed codec bundle (one template bioassembly per trajectory + a stacked-coordinate array, reconstructed losslessly on the fly). See **[docs/data_codec.md](docs/data_codec.md)**.

The trained checkpoints and the codec bundle are hosted on HuggingFace: <https://huggingface.co/fengb/BioKinema>.

## Reproducing the Manuscript

The `experiments/` directory provides self-contained scripts and expected metrics for the main results: the Atlas kinetics benchmark, the protein–ligand case studies (ADK, Pin1), and the long-time kinetics/thermodynamics evaluation. See **[experiments/README.md](experiments/README.md)**.

## Citation

```bibtex
@article{feng2026physically,
  title={Physically Grounded Generative Modeling of All-Atom Biomolecular Dynamics},
  author={Feng, Bin and Zhang, Jiying and Zhang, Xinni and Zhang, Ming and Barth, Patrick and Liu, Zijing and Li, Yu},
  journal={bioRxiv},
  pages={2026--02},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Acknowledgements

This project was built based on [Protenix](https://github.com/bytedance/Protenix), an open-source biomolecular structure prediction framework developed by ByteDance.

## Contact

For questions or collaborations, please open an issue or contact us at [fengbin@idea.edu.cn](mailto:fengbin@idea.edu.cn).
