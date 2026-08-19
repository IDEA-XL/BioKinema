# Atlas & MSR data preparation + MSM caches

For Atlas and the MSR datasets (MDCATH / octapeptides) the preprocessing is
simple enough to publish directly. The pipeline turns raw MD trajectories into the
per-frame `bioassembly_dict` pickles + index CSVs the training code consumes.

## Atlas

```bash
# 1. download raw Atlas trajectories
bash scripts/data_prep/download_atlas.sh

# 2. trajectory frames (xtc + topology) -> per-frame mmCIF (0.1 ns interval)
python scripts/data_prep/convert_xtc_to_cif.py \
    --atlas_dir /raw/atlas --outdir $BIOKINEMA_ATLAS_ROOT --num_workers 32

# 3. mmCIF -> bioassembly pickles + index CSV
python scripts/data_prep/prepare_training_data.py \
    -i $BIOKINEMA_ATLAS_ROOT/mmcif \
    -o $BIOKINEMA_ATLAS_ROOT/indices.csv \
    -b $BIOKINEMA_ATLAS_ROOT/mmcif_bioassembly \
    -d Atlas -n 32
```

## MSR (BioEmu-format: MDCATH, octapeptides)

```bash
# frames -> per-frame mmCIF (10 ns interval), grouped by system
python scripts/data_prep/convert_xtc_to_cif_bioemu.py \
    --bioemu_dir /raw/mdcath --outdir $BIOKINEMA_MSR_ROOT/MDCATH/MSR_cath2_biokinema \
    --num_workers 32

# mmCIF -> bioassembly pickles + per-system CSVs
python scripts/data_prep/prepare_training_data_bydir.py \
    -i  $BIOKINEMA_MSR_ROOT/MDCATH/MSR_cath2_biokinema/mmcif \
    -o  $BIOKINEMA_MSR_ROOT/MDCATH/MSR_cath2_biokinema/csv \
    -b  $BIOKINEMA_MSR_ROOT/MDCATH/MSR_cath2_biokinema/bio \
    -n 32
```

## MSM caches (TICA-dynamics loss)

The TICA-dynamics loss (used by the `beta=0.25` / MSR training) reads prebuilt MSM caches.
`configs/configs_data.py` points each MSR dataset at
`$BIOKINEMA_MSM_CACHES/<name>_lag10ns_from100ns_multiK`. Build them from the bioassembly data:

```bash
python scripts/msm/build_multiK_msm.py \
    --dataset MSR \
    --bio-dir  $BIOKINEMA_MSR_ROOT/MDCATH/MSR_cath2_biokinema/bio \
    --csv-dir  $BIOKINEMA_MSR_ROOT/MDCATH/MSR_cath2_biokinema/csv \
    --out-dir  $BIOKINEMA_MSM_CACHES/CATH2_lag10ns_from100ns_multiK \
    --k-values 10,20,50 --max-gmm-components 5 --n-workers 64
# Repeat per MSR dataset (CATH1 / octapeptide).
```

Each cache stores, per system: a TICA basis (Cα–Cα pairwise distances, 10 ns lag, 5 components),
multi-K coarse MSMs (transition matrix estimated at a 100 ns counting lag, then matrix-rooted to a
10 ns operator; with stationary distribution), and per-state diagonal-Gaussian emissions.
