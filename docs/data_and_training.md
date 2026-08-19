# Data download, preprocessing & training

This document covers, end to end: **(A)** which datasets BioKinema trains on and how to obtain
them, **(B)** how to turn raw MD into the training format, **(C)** the precomputed embeddings,
and **(D)** how to launch training. Inference/benchmarks are documented separately (see
`docs/inference.md` and `experiments/`).

> Download commands for every public source are inlined below. Our own artifacts — the two
> checkpoints and the MISATO/MDposit/unbinding codec bundle — are on HuggingFace:
> <https://huggingface.co/fengb/BioKinema>.

---

## 0. The two released checkpoints

BioKinema releases two checkpoints (download: <https://huggingface.co/fengb/BioKinema>):

| Checkpoint | For | `beta` | Training data | Active dynamics loss |
|---|---|---|---|---|
| **`sqrt`** | complexes (protein–ligand) + **short-time** MD | 0.5 | Atlas + MISATO + MDposit (+ unbinding) | RMSF / RelRMSF / LocalRMSF / ACF / ensemble |
| **`beta=0.25`** | **long-time, single-chain** protein MD | 0.25 | MSR (CATH2 / octapeptides) | + TICA-dynamics (needs MSM caches) |

Both are produced by the **same** trajectory-generation training (`train_trajectory.sh`),
differing only in the training data and `β`.

---

## 1. Datasets at a glance

| Dataset | Used by | Public raw source | How you get the training-ready data |
|---|---|---|---|
| **Base PDB / CCD** (`BIOKINEMA_DATA_ROOT`) | both | `https://af3-dev.tos-cn-beijing.volces.com/release_data.tar.gz` | `wget` + untar into `$BIOKINEMA_DATA_ROOT` |
| **Atlas** (`BIOKINEMA_ATLAS_ROOT`) | `sqrt` | ATLAS DB (public) | `scripts/data_prep/download_atlas.sh` → `convert_xtc_to_cif.py` → `prepare_training_data.py` |
| **MSR: mdCATH / octapeptides** (`BIOKINEMA_MSR_ROOT`) | `beta=0.25` | Zenodo: CATH `10.5281/zenodo.15629740`, Octapeptides `…15641199` | `convert_xtc_to_cif_bioemu.py` → `prepare_training_data_bydir.py` → `scripts/msm` |
| **MISATO / MDposit / unbinding** (`BIOKINEMA_UNBINDING_ROOT`) | `sqrt` | preprocessing too source-specific to publish | **download our compressed pre-processed data** (codec; see §4) — no raw download needed |
| **MSM caches** (`BIOKINEMA_MSM_CACHES`) | `beta=0.25` (TICA loss) | — | build from the MSR data with `scripts/msm` (§3) |

Atlas & MSR are released as **preprocessing recipes** (you download public raw MD and run the
scripts). MISATO/MDposit/unbinding are released as **compressed processed data** (the codec
bundle), because their raw→bioassembly preprocessing is not portable.

---

## 2. Atlas (recipe)

```bash
export BIOKINEMA_ATLAS_ROOT=/path/to/atlas

# (1) download raw Atlas trajectories for a split (uses scripts/splits/atlas_{train,val,test}.csv)
bash scripts/data_prep/download_atlas.sh $BIOKINEMA_ATLAS_ROOT/raw atlas_train

# (2) raw xtc+pdb -> per-frame mmCIF (0.1 ns interval)
python scripts/data_prep/convert_xtc_to_cif.py \
    --atlas_dir $BIOKINEMA_ATLAS_ROOT/raw --outdir $BIOKINEMA_ATLAS_ROOT --num_workers 32

# (3) mmCIF -> bioassembly pickles + index CSV
python scripts/data_prep/prepare_training_data.py \
    -i $BIOKINEMA_ATLAS_ROOT/mmcif -o $BIOKINEMA_ATLAS_ROOT/indices.csv \
    -b $BIOKINEMA_ATLAS_ROOT/mmcif_bioassembly -d Atlas -n 32
```
The 1266/39/81 train/val/test system lists are in `scripts/splits/` (regenerated from the
released indices; `test` = the 81 benchmark targets).

## 3. MSR — mdCATH / octapeptides (recipe)

```bash
export BIOKINEMA_MSR_ROOT=/path/to/MSR
ROOT=$BIOKINEMA_MSR_ROOT/MDCATH/MSR_cath2_biokinema   # one dataset; repeat per dataset

# (1) download raw MD from Zenodo (CATH 10.5281/zenodo.15629740,
#     Octapeptides 10.5281/zenodo.15641199) and extract, then:
# (2) frames -> per-frame mmCIF (10 ns interval), grouped by system
python scripts/data_prep/convert_xtc_to_cif_bioemu.py --bioemu_dir /raw/mdcath --outdir $ROOT --num_workers 32
# (3) mmCIF -> bioassembly pickles + per-system CSVs
python scripts/data_prep/prepare_training_data_bydir.py -i $ROOT/mmcif -o $ROOT/csv -b $ROOT/bio -n 32
# (4) MSM caches for the TICA-dynamics loss (single step; builds the final
#     `*_lag10ns_from100ns_multiK` operator = reversible count @ 100 ns, root ^(1/10) → 10 ns):
python scripts/msm/build_multiK_msm.py --dataset MSR --bio-dir $ROOT/bio --csv-dir $ROOT/csv \
    --out-dir $BIOKINEMA_MSM_CACHES/CATH2_lag10ns_from100ns_multiK --k-values 10,20,50 --n-workers 64
```
See `docs/data_msm.md` for the MSM-cache contents and the lag-conversion rationale.

## 4. MISATO / MDposit / unbinding — compressed processed data

These ship **pre-processed and compressed** (one template bioassembly per trajectory + a
stacked-coordinate array; lossless, ~3× smaller). Download the codec bundle and point the
dataset root at it — the dataset auto-detects and decompresses on the fly (no extra step):

```bash
# download + extract the codec bundle (from https://huggingface.co/fengb/BioKinema) so that:
#   $BIOKINEMA_UNBINDING_ROOT/misato_codec/<traj>.tpl.pkl.gz + <traj>.coords.npz   (etc.)
export BIOKINEMA_UNBINDING_ROOT=/path/to/biokinema_processed
```
Point each dataset's `bioassembly_dict_dir` (in `configs/configs_data.py`) at the corresponding
`*_codec/` dir. See `docs/data_codec.md` for the format and how to (re)compress your own data.

## 5. Precomputed embeddings (both checkpoints)

Every training config reads Pairformer embeddings from each dataset's `precomputed_emb_dir`
instead of recomputing them each step. Regenerate them from a checkpoint (we do not ship the
multi-TB embeddings):
```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i bash scripts/encode_embeddings.sh <dataset> $i 8 &
done; wait      # e.g. <dataset> = atlas_train, misato_train, MSR-CATH2, ...
```

The encoder writes one `<traj_name>.pt` per system into that dataset's `precomputed_emb_dir`
(resolved from `configs/configs_data.py`) — the same path training reads from.

## 6. Training

Data roots are set via environment variables (resolved in `configs/configs_data.py`):

| env var | content |
|---|---|
| `BIOKINEMA_DATA_ROOT` | base PDB/CCD release data (`components.cif`, MSA index, …) |
| `BIOKINEMA_ATLAS_ROOT` | Atlas mmcif / bioassembly / indices |
| `BIOKINEMA_UNBINDING_ROOT` (= MISATO root) | MISATO / MDposit / unbinding (codec bundle) |
| `BIOKINEMA_MSR_ROOT` | mdCATH / octapeptides |
| `BIOKINEMA_MSM_CACHES` | prebuilt MSM caches (TICA-dynamics loss) |
| `BIOKINEMA_INIT_CKPT` | checkpoint to initialize training from |

**`sqrt` model** (complexes + short MD):
```bash
export BIOKINEMA_INIT_CKPT=/path/to/init.pt   # base Protenix or a prior BioKinema ckpt
NPROC_PER_NODE=8 bash train_trajectory.sh
```
(`train_trajectory.sh` defaults: `train_sets=misato_train,atlas_train,mdposit`,
`beta=0.5`, `alpha_tica_dynamics=0.25` — inactive unless a dataset has an MSM cache.)

**`beta=0.25` model** (long single-chain MD) — same script, recipe set via env vars:
```bash
export BIOKINEMA_INIT_CKPT=/path/to/init.pt
export BIOKINEMA_MSR_ROOT=/path/to/MSR  BIOKINEMA_MSM_CACHES=/path/to/msm_caches
NPROC_PER_NODE=8 \
RUN_NAME=BioKinema_beta0.25 \
TRAIN_SETS=MSR-CATH2,MSR-CATH1,MSR-octapeptide \
TEST_SETS=MSR-CATH2 \
SAMPLE_WEIGHTS=0.756,0.095,0.148 \
BETA=0.25 \
bash train_trajectory.sh
```
(MSR datasets carry MSM caches, so the TICA-dynamics loss is active here.)

Overridable env vars: `RUN_NAME`, `TRAIN_SETS`, `TEST_SETS`, `SAMPLE_WEIGHTS`, `BETA`,
`EVAL_FIRST`, `USE_WANDB` (default false), `NPROC_PER_NODE`, `MASTER_PORT`. Extra `--flag value`
args are forwarded to `runner/train.py`. On resume also pass `--load_ema_checkpoint_path`.

---

## Data sources (summary)

| Need | Where |
|---|---|
| Base PDB / CCD (`release_data`) + init checkpoint | `https://af3-dev.tos-cn-beijing.volces.com/{release_data.tar.gz, release_model/model_v0.2.0.pt}` |
| Atlas | `scripts/data_prep/download_atlas.sh` (§2) |
| MSR raw MD: CATH / Octapeptides | Zenodo `10.5281/zenodo.{15629740, 15641199}` |
| MISATO / MDposit / unbinding codec bundle | <https://huggingface.co/fengb/BioKinema> |
| Checkpoints (`sqrt`, `beta=0.25`) | <https://huggingface.co/fengb/BioKinema> |
| MSM caches | built from the MSR data via `scripts/msm` (§3) |
