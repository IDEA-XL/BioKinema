# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0114,C0301
import os
from copy import deepcopy

from protenix.config.extend_types import GlobalConfigValue, ListValue

# Data roots are resolved from environment variables so the open-source repo is
# portable. Point these at wherever you placed the processed training data:
#   BIOKINEMA_DATA_ROOT      -> base PDB/CCD release data (components.cif, mmcif, msa, ...)
#   BIOKINEMA_ATLAS_ROOT     -> Atlas mmcif/bioassembly/indices (see scripts/data_prep)
#   BIOKINEMA_UNBINDING_ROOT -> unbinding/MISATO/MDposit processed data (see scripts/codec)
#   BIOKINEMA_MISATO_ROOT    -> defaults to BIOKINEMA_UNBINDING_ROOT
#   BIOKINEMA_MSM_CACHES     -> prebuilt MSM caches (see scripts/msm)
#   BIOKINEMA_MSR_ROOT       -> BioEmu/MSR datasets (CATH1/CATH2/octapeptides)
DATA_ROOT_DIR = os.environ.get("BIOKINEMA_DATA_ROOT", "./release_data")
ATLAS_DATA_ROOT_DIR = os.environ.get("BIOKINEMA_ATLAS_ROOT", "./data/atlas")
UNBINDING_DATA_ROOT_DIR = os.environ.get("BIOKINEMA_UNBINDING_ROOT", "./data")

MISATO_DATA_ROOT_DIR = os.environ.get("BIOKINEMA_MISATO_ROOT", UNBINDING_DATA_ROOT_DIR)

default_test_configs = {
    "sampler_configs": {
        "sampler_type": "uniform",
    },
    "cropping_configs": {
        "method_weights": [
            0.0,  # ContiguousCropping
            0.0,  # SpatialCropping
            1.0,  # SpatialInterfaceCropping
        ],
        "crop_size": -1,
    },
    "lig_atom_rename": GlobalConfigValue("test_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("test_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("test_shuffle_sym_ids"),
}

default_weighted_pdb_configs = {
    "sampler_configs": {
        "sampler_type": "weighted",
        "beta_dict": {
            "chain": 0.5,
            "interface": 1,
        },
        "alpha_dict": {
            "prot": 3,
            "nuc": 3,
            "ligand": 1,
        },
        "force_recompute_weight": True,
    },
    "cropping_configs": {
        "method_weights": ListValue([0.2, 0.4, 0.4]),
        "crop_size": GlobalConfigValue("train_crop_size"),
    },
    "sample_weight": 0.5,
    "limits": -1,
    "lig_atom_rename": GlobalConfigValue("train_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("train_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("train_shuffle_sym_ids"),
}


# Use CCD cache created by scripts/gen_ccd_cache.py priority. (without date in filename)
# See: docs/prepare_data.md
CCD_COMPONENTS_FILE_PATH = os.path.join(DATA_ROOT_DIR, "components.cif")
CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
    DATA_ROOT_DIR, "components.cif.rdkit_mol.pkl"
)

if (not os.path.exists(CCD_COMPONENTS_FILE_PATH)) or (
    not os.path.exists(CCD_COMPONENTS_RDKIT_MOL_FILE_PATH)
):
    CCD_COMPONENTS_FILE_PATH = os.path.join(DATA_ROOT_DIR, "components.v20240608.cif")
    CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
        DATA_ROOT_DIR, "components.v20240608.cif.rdkit_mol.pkl"
    )


# This is a patch in inference stage for users that do not have root permission.
# If you run
# ```
# bash inference_demo.sh
# ```
# or
# ```
# protenix predict --input examples/example.json --out_dir  ./output
# ````
# The checkpoint and the data cache will be downloaded to the current code directory.
if (not os.path.exists(CCD_COMPONENTS_FILE_PATH)) or (
    not os.path.exists(CCD_COMPONENTS_RDKIT_MOL_FILE_PATH)
):
    print("Try to find the ccd cache data in the code directory for inference.")
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    code_directory = os.path.dirname(current_directory)

    data_cache_dir = os.path.join(code_directory, "release_data/ccd_cache")
    CCD_COMPONENTS_FILE_PATH = os.path.join(data_cache_dir, "components.cif")
    CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
        data_cache_dir, "components.cif.rdkit_mol.pkl"
    )
    if (not os.path.exists(CCD_COMPONENTS_FILE_PATH)) or (
        not os.path.exists(CCD_COMPONENTS_RDKIT_MOL_FILE_PATH)
    ):

        CCD_COMPONENTS_FILE_PATH = os.path.join(
            data_cache_dir, "components.v20240608.cif"
        )
        CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
            data_cache_dir, "components.v20240608.cif.rdkit_mol.pkl"
        )

data_configs = {
    "num_dl_workers": 16,
    "epoch_size": 10000,
    "train_ref_pos_augment": True,
    "test_ref_pos_augment": True,
    "train_sets": ListValue(["weightedPDB_before2109_wopb_nometalc_0925"]),
    "train_sampler": {
        "train_sample_weights": ListValue([1.0]),
        "sampler_type": "weighted",
    },
    "test_sets": ListValue(["recentPDB_1536_sample384_0925"]),
    "weightedPDB_before2109_wopb_nometalc_0925": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(DATA_ROOT_DIR, "mmcif_bioassembly"),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR,
                "indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz",
            ),
            "pdb_list": "",
            "random_sample_if_failed": True,
            "max_n_token": -1,  # can be used for removing data with too many tokens.
            "use_reference_chains_only": False,
            "exclusion": {  # do not sample the data based on ions.
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
        },
        **deepcopy(default_weighted_pdb_configs),
    },
    "recentPDB_1536_sample384_0925": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "mmcif"),
            "bioassembly_dict_dir": os.path.join(
                DATA_ROOT_DIR, "recentPDB_bioassembly"
            ),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR, "indices/recentPDB_low_homology_maxtoken1536.csv"
            ),
            "pdb_list": os.path.join(
                DATA_ROOT_DIR,
                "indices/recentPDB_low_homology_maxtoken1024_sample384_pdb_id.txt",
            ),
            "max_n_token": GlobalConfigValue("test_max_n_token"),  # filter data
            "sort_by_n_token": False,
            "group_by_pdb_id": True,
            "find_eval_chain_interface": True,
        },
        **deepcopy(default_test_configs),
    },
    "posebusters_0925": {
        "base_info": {
            "mmcif_dir": os.path.join(DATA_ROOT_DIR, "posebusters_mmcif"),
            "bioassembly_dict_dir": os.path.join(
                DATA_ROOT_DIR, "posebusters_bioassembly"
            ),
            "indices_fpath": os.path.join(
                DATA_ROOT_DIR, "indices/posebusters_indices_mainchain_interface.csv"
            ),
            "pdb_list": "",
            "find_pocket": True,
            "find_all_pockets": False,
            "max_n_token": GlobalConfigValue("test_max_n_token"),  # filter data
        },
        **deepcopy(default_test_configs),
    },
    "msa": {
        "enable": True,
        "enable_rna_msa": False,
        "prot": {
            "pairing_db": "uniref100",
            "non_pairing_db": "mmseqs_other",
            "pdb_mmseqs_dir": os.path.join(DATA_ROOT_DIR, "mmcif_msa"),
            "seq_to_pdb_idx_path": os.path.join(DATA_ROOT_DIR, "seq_to_pdb_index.json"),
            "indexing_method": "sequence",
        },
        "rna": {
            "seq_to_pdb_idx_path": "",
            "rna_msa_dir": "",
            "indexing_method": "sequence",
        },
        "strategy": "random",
        "merge_method": "dense_max",
        "min_size": {
            "train": 1,
            "test": 2048,
        },
        "max_size": {
            "train": 16384,
            "test": 16384,
        },
        "sample_cutoff": {
            "train": 2048,
            "test": 2048,
        },
    },
    "template": {
        "enable": False,
    },
    "ccd_components_file": CCD_COMPONENTS_FILE_PATH,
    "ccd_components_rdkit_mol_file": CCD_COMPONENTS_RDKIT_MOL_FILE_PATH,
}

########################
# atlas finetuning data
########################
data_configs["atlas"] = {
    "base_info": {
        "mmcif_dir": os.path.join(ATLAS_DATA_ROOT_DIR, "mmcif"),
        "bioassembly_dict_dir": os.path.join(ATLAS_DATA_ROOT_DIR, "mmcif_bioassembly"),
        "indices_fpath": os.path.join(
            ATLAS_DATA_ROOT_DIR, "indices.csv",
        ),
        # keep the belows as is
        "pdb_list": "",
        "random_sample_if_failed": True,
        "max_n_token": -1,  # can be used for removing data with too many tokens.
        "use_reference_chains_only": True,
    },
        **deepcopy(default_weighted_pdb_configs),
}
data_configs["atlas"].update(
    {
        "sampler_configs": {
            "sampler_type": "uniform",
        },
        "cropping_configs": {
            "method_weights": ListValue([0.0, 1.0, 0.0]),
            "crop_size": GlobalConfigValue("train_crop_size"),
        },
        "precomputed_emb_dir": os.path.join(ATLAS_DATA_ROOT_DIR, "pairformer_emb"),
        "msa_cache_path": os.path.join(ATLAS_DATA_ROOT_DIR, "msa"),
        # MSM for TICA dynamics loss — DISABLED for Atlas (lag=1ns picks inter-replica drift,
        # produces clusters scattered across disjoint replica subspaces). Sentinel "NONE"
        # → dataset.py skips MSM init → has_msm=False → tica_dynamics_loss skipped.
        "msm_cache_dir": "NONE",
        "msm_build_missing": False,  # pre-built cache: skip uncached systems rather than building from bio files
        "n_coarse_states": 10,  # K to select from multi-K dict format
        "msm_configs": {
            "n_tica_dims": 5,
            "n_clusters_coarse": 10,
            "n_clusters_fine": 50,
        },
    }
)

for split in ['train', 'val', 'test', 'repr', 'test_repr']:
    data_configs[f"atlas_{split}"] = deepcopy(data_configs["atlas"])
    # if split != 'train':
    data_configs[f"atlas_{split}"].update(
        **deepcopy(default_test_configs),
    )
    data_configs[f"atlas_{split}"]["base_info"]["indices_fpath"] = os.path.join(
        ATLAS_DATA_ROOT_DIR, f"indices_{split}.csv"
    )

####################################
# example for new dataset template
####################################
# data_configs[f"example"] = deepcopy(data_configs["atlas"])
# data_configs[f"example"]["base_info"].update(
#     {
#         "mmcif_dir": "./example/mmcif",  # use the example mmcif dir
#         "bioassembly_dict_dir": "./example/bioassembly",  # use the example bioassembly dir
#         "indices_fpath": "./example/indices.csv",  # use the example indices file
#     }
# )
# data_configs[f"example"].update(
#      {   
#          "precomputed_emb_dir": "/cto_studio/xtalpi_lab/fengbin/Protenix_v0.2.0/Protenix/fb_analysis/completed_embed_files"
#      }
# )


# data_configs[f"example_test"] = deepcopy(data_configs["atlas"])
# data_configs[f"example_test"].update(
#     **deepcopy(default_test_configs),
# )
# data_configs[f"example_test"]["base_info"].update(
#     {
#         "mmcif_dir": "./fb_analysis/processed_cif_files",  # use the example mmcif dir
#         "bioassembly_dict_dir": "./fb_analysis/processed_cif_files_bioassembly",  # use the example bioassembly dir
#         "indices_fpath": "./fb_analysis/indices_processed_cif_files.csv",  # use the example indices file
#     }
# )
# data_configs[f"example_test"].update(
#      {   
#          "precomputed_emb_dir": ""
#      }
# )


data_configs[f"example"] = deepcopy(data_configs["atlas"])
data_configs[f"example"].update(
    **deepcopy(default_test_configs),
)
data_configs[f"example"]["base_info"].update(
    {
        "mmcif_dir": "./fb_analysis/completed_cif_files_bydir",  # use the example mmcif dir
        "bioassembly_dict_dir": "./fb_analysis/completed_bio_files_bydir",  # use the example bioassembly dir
        "indices_fpath": "./fb_analysis/completed_csv_files_bydir",  # use the example indices file
        "use_reference_chains_only": False,
    }
)
data_configs[f"example"].update(
     {   
         "precomputed_emb_dir": ""
     }
)


data_configs[f"inference"] = deepcopy(data_configs["atlas"])
data_configs[f"inference"].update(
    **deepcopy(default_test_configs),
)
data_configs[f"inference"]["base_info"].update(
    {
        "use_reference_chains_only": False,
    }
)
# Inference computes embeddings on-the-fly; no precomputed embedding dir.
data_configs[f"inference"]["precomputed_emb_dir"] = ""
# MSA cache for inference: sequences missing from the precomputed index are searched online
# and cached here (msa_featurizer.msa_search reuses an existing hit instead of re-searching).
# Inference-only on purpose: inherited from atlas this would land in the Atlas *training* data
# dir, which an inference-only user does not have. Training datasets keep their own paths.
data_configs[f"inference"]["msa_cache_path"] = os.environ.get(
    "BIOKINEMA_MSA_CACHE_DIR", os.path.join(DATA_ROOT_DIR, "msa")
)


bioemu_root = os.path.join(UNBINDING_DATA_ROOT_DIR, "bioemu")
for testset in ['crypticpocket', 'domainmotion', 'localunfolding', 'ood60', 'oodval']:
    data_configs[f"bioemu-{testset}"] = deepcopy(data_configs["atlas"])
    data_configs[f"bioemu-{testset}"].update(
        **deepcopy(default_test_configs),
    )
    data_configs[f"bioemu-{testset}"]["base_info"].update(
        {
            "mmcif_dir": f"{bioemu_root}/{testset}/mmcif",  
            "bioassembly_dict_dir": f"{bioemu_root}/{testset}/bio",  
            "indices_fpath": f"{bioemu_root}/{testset}/csv",  
            "use_reference_chains_only": False,
        }
    )
    data_configs[f"bioemu-{testset}"].update(
        {   
            "precomputed_emb_dir": f"{bioemu_root}/embed_files"
        }
    )


####################################
# unbinding finetuning data
####################################
data_configs[f"unbinding"] = deepcopy(data_configs["atlas"])
data_configs[f"unbinding"]["base_info"].update(
    {
        "mmcif_dir": os.path.join(UNBINDING_DATA_ROOT_DIR, "unbinding_cif"),  # use the unbinding mmcif dir
        "bioassembly_dict_dir": os.path.join(UNBINDING_DATA_ROOT_DIR, "unbinding_bio"),  # use the unbinding bioassembly dir
        "indices_fpath": os.path.join(UNBINDING_DATA_ROOT_DIR, "unbinding_csv", "all"),
        "use_reference_chains_only": False,
        "random_sample_if_failed": True,
    }
)
data_configs[f"unbinding"].update(
    {
        "sampler_configs": {
            "sampler_type": "uniform",
        },
        "cropping_configs": {
            "method_weights": ListValue([0.0, 1.0, 0.0]),
            "crop_size": GlobalConfigValue("train_crop_size"),
        },
        # "precomputed_emb_dir": 
        # "precomputed_emb_dir": "NONE",
        "precomputed_emb_dir": os.path.join(UNBINDING_DATA_ROOT_DIR, "embed_unbinding_nomsa")
    }
)

for split in ['train', 'val', 'test']:
    data_configs[f"unbinding_{split}"] = deepcopy(data_configs["unbinding"])
    # if split != 'train':
    data_configs[f"unbinding_{split}"].update(
        **deepcopy(default_test_configs),
    )
    if split == "train":
        data_configs[f"unbinding_{split}"]["base_info"]["indices_fpath"] = os.path.join(UNBINDING_DATA_ROOT_DIR, "unbinding_csv", split)
    else:
        data_configs[f"unbinding_{split}"]["base_info"]["indices_fpath"] = os.path.join(UNBINDING_DATA_ROOT_DIR, "unbinding_csv", split+'_onetraj')


####################################
# misato finetuning data
####################################
data_configs[f"misato"] = deepcopy(data_configs["atlas"])
data_configs[f"misato"]["base_info"].update(
    {
        "mmcif_dir": os.path.join(MISATO_DATA_ROOT_DIR, "misato_mmcif_fixed"),  # use the misato mmcif dir
        "bioassembly_dict_dir": os.path.join(MISATO_DATA_ROOT_DIR, "misato_bio_noref"),  # use the misato bioassembly dir
        "indices_fpath": os.path.join(MISATO_DATA_ROOT_DIR, "misato_csv_noref", "all"),  # use the misato indices file
        "use_reference_chains_only": False,
        "random_sample_if_failed": True,
    }
)
data_configs["misato"].update(
    {
        "sampler_configs": {
            "sampler_type": "uniform",
        },
        "cropping_configs": {
            "method_weights": ListValue([0.0, 1.0, 0.0]),
            "crop_size": GlobalConfigValue("train_crop_size"),
        },
        # "precomputed_emb_dir": 
        # "precomputed_emb_dir": "NONE",
        "precomputed_emb_dir": os.path.join(MISATO_DATA_ROOT_DIR, "embed_misato_msa_noref"),
    }
)
data_configs["misato"].update(
    **deepcopy(default_test_configs),
)

for split in ['train', 'val', 'test', 'teststrict']:
    data_configs[f"misato_{split}"] = deepcopy(data_configs["misato"])
    # if split != 'train':
    data_configs[f"misato_{split}"].update(
        **deepcopy(default_test_configs),
    )
    data_configs[f"misato_{split}"]["base_info"]["indices_fpath"] = os.path.join(MISATO_DATA_ROOT_DIR, "misato_csv_noref", split)


####################################
# mdposit finetuning data
####################################
data_configs[f"mdposit"] = deepcopy(data_configs["atlas"])
data_configs[f"mdposit"]["base_info"].update(
    {
        "mmcif_dir": os.path.join(MISATO_DATA_ROOT_DIR, "MDposit_mmcif_fixed_full"),  # use the mdposit mmcif dir
        "bioassembly_dict_dir": os.path.join(MISATO_DATA_ROOT_DIR, "MDposit_bio_noref"),  # use the mdposit bioassembly dir
        "indices_fpath": os.path.join(MISATO_DATA_ROOT_DIR, "MDposit_csv_noref"),  # use the mdposit indices file
        "use_reference_chains_only": False,
        "random_sample_if_failed": True
    }
)
data_configs["mdposit"].update(
    {
        "sampler_configs": {
            "sampler_type": "uniform",
        },
        "cropping_configs": {
            "method_weights": ListValue([0.0, 1.0, 0.0]),
            "crop_size": GlobalConfigValue("train_crop_size"),
        },
        # "precomputed_emb_dir": 
        # "precomputed_emb_dir": "NONE",
        "precomputed_emb_dir": os.path.join(MISATO_DATA_ROOT_DIR, "embed_mdposit_noref"),
        "interval_fpath": os.path.join(MISATO_DATA_ROOT_DIR, "MDposit_time_interval.json"),
    }
)


# for split in ['train', 'val', 'test']:
#     data_configs[f"mdposit_{split}"] = deepcopy(data_configs["mdposit"])
#     # if split != 'train':
#     # data_configs[f"mdposit_{split}"].update(
#     #     **deepcopy(default_test_configs),
#     # )
#     data_configs[f"mdposit_{split}"]["base_info"]["indices_fpath"] = os.path.join(MISATO_DATA_ROOT_DIR, "MDposit_csv_full", split)


####################################
# BioEmu finetuning data
####################################
# MSR / BioEmu datasets. Root resolved from BIOKINEMA_MSR_ROOT; subdirs follow
# scripts/data_prep (convert_xtc_to_cif_bioemu.py + prepare_training_data_bydir.py).
MSR_ROOT = os.environ.get("BIOKINEMA_MSR_ROOT", "./data/MSR")
data_root_dict = {
    "CATH2": os.path.join(MSR_ROOT, "MDCATH/MSR_cath2_biokinema"),
    "CATH1": os.path.join(MSR_ROOT, "MDCATH/ONE_cath1_biokinema"),
    "octapeptide": os.path.join(MSR_ROOT, "octapeptides/ONE_octapeptides_biokinema"),
    }

for data_name, data_root in data_root_dict.items():
    data_configs[f"MSR-{data_name}"] = deepcopy(data_configs["atlas"])
    data_configs[f"MSR-{data_name}"]["base_info"].update(
        {
            "mmcif_dir": os.path.join(data_root, "mmcif"),
            "bioassembly_dict_dir": os.path.join(data_root, "bio"),
            "indices_fpath": os.path.join(data_root, "csv"),  # all systems
            "use_reference_chains_only": False,
            "random_sample_if_failed": True
        }
    )

    # NOTE: directories are created by the data-prep / encode scripts, not at
    # config-import time (importing a config must have no filesystem side effects).
    embed_dir = os.path.join(data_root, "embed")
    msa_cache_path = os.path.join(data_root, "msa")

    # BioEmu: 10ns/frame.
    # Cache "{data_name}_lag10ns_from100ns_multiK": saved T is at effective lag=10ns
    # (lagtime_frames=1) but constructed via reversible MSM at counting-lag=100ns
    # (10 frames) followed by matrix root ^(1/10). This captures slow modes much
    # better than directly estimating at lag=10ns while remaining compatible with
    # downstream loss code that interprets the saved T as a lag=10ns transition.
    msm_cache_dir = os.path.join(
        os.environ.get("BIOKINEMA_MSM_CACHES", "./data/MSM_CACHES"),
        f"{data_name}_lag10ns_from100ns_multiK",
    )

    data_configs[f"MSR-{data_name}"].update(
        {
            "sampler_configs": {
                "sampler_type": "uniform",
            },
            "cropping_configs": {
                "method_weights": ListValue([0.0, 1.0, 0.0]),
                "crop_size": GlobalConfigValue("train_crop_size"),
            },
            # "precomputed_emb_dir":
            # "precomputed_emb_dir": "NONE",
            "precomputed_emb_dir": embed_dir,
            "msa_cache_path": msa_cache_path,
            # MSM for TICA dynamics loss + pseudo-trajectory augmentation
            # lag=10ns (1 frame @ 10ns/frame) — physically motivated, CK-validated
            "msm_cache_dir": msm_cache_dir,
            "n_coarse_states": 10,  # K used for loss-side artifacts (cluster centers,
                                    # log_vars, stationary distribution → population KL).
                                    # KL is N/K limited; with N=64 frame_t halves, K=10 is
                                    # the practical upper bound.
            "n_sampling_states": 50, # K used for MSM-mode state-pair sampling
                                    # (s_0 ~ π_K, s_t ~ T_K^k[s_0,:]). Larger K gives finer
                                    # dynamics signal per pair. Decoupled from loss K so we
                                    # can have rich sampling without harming KL statistics.
                                    # In non-MSM modes this is ignored.
            "msm_configs": {
                "n_tica_dims": 5,
                "n_clusters_coarse": 10,
                "n_clusters_fine": 50,
            },
        }
    )

    # Per-split variants: csv/{train,val,test}/ subdirs (symlinks to per-system CSVs)
    # train: 1000 systems, val: 20 systems, test: 20 systems
    for split in ["train", "val", "test"]:
        key = f"MSR-{data_name}-{split}"
        data_configs[key] = deepcopy(data_configs[f"MSR-{data_name}"])
        data_configs[key]["base_info"]["indices_fpath"] = os.path.join(data_root, "csv", split)

