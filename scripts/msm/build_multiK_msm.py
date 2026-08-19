"""
build_multiK_msm.py — Build the MSM caches used by the TICA-dynamics loss.

For each system: fit a TICA basis from the bioassembly Cα coordinates, build coarse MSMs at
K=10,20,50 (configurable), and fit per-state diagonal-Gaussian (GMM) emissions. The transition
matrix is estimated at a 100 ns counting lag and matrix-rooted to a 10 ns operator. Output is one
plain-dict PKL per system, named to match the `{NAME}_lag10ns_from100ns_multiK` dir configs_data.py
reads. (Pass --cache-dir/--gmm-cache-dir to reuse a precomputed TICA basis instead of fitting it.)

PKL format:
{
  "tica_mean": [N_pairs],
  "tica_components": [N_pairs, N_tica],
  "tica_eigenvalues": [N_tica],
  "pair_indices_i": [N_pairs],
  "pair_indices_j": [N_pairs],
  "n_tica_dims": int,
  "lagtime_frames": int,
  "n_ca_full": int,
  "traj_frame_counts": {traj: int},
  "tica_coords_by_traj": {traj: [T, N_tica]},   # always stored
  "by_k": {
      K: {
          "n_states": int,
          "cluster_centers": [K, N_tica],
          "state_labels": {traj: [T]},            # -1 for disconnected
          "transition_matrix": [K, K],
          "stationary_dist": [K],
          "gmm_by_c": {
              C: {"weights": [K,C], "means": [K,C,NT], "precisions_chol": [K,C,NT,NT]}
          }
      }
  }
}

Usage (release recipe — MSR datasets, from raw bioassembly data):
  python scripts/msm/build_multiK_msm.py \\
      --dataset MSR \\
      --bio-dir  $BIOKINEMA_MSR_ROOT/MDCATH/MSR_cath2_biokinema/bio \\
      --csv-dir  $BIOKINEMA_MSR_ROOT/MDCATH/MSR_cath2_biokinema/csv \\
      --out-dir  $BIOKINEMA_MSM_CACHES/CATH2_lag10ns_from100ns_multiK \\
      --k-values 10,20,50 \\
      --max-gmm-components 5 \\
      --n-workers 64
  # (TICA lag defaults: --n-tica-dims 5, --tica-lagtime-frames 1 = 10 ns for MSR.)
  # Repeat per MSR dataset (CATH1 / CATH2 / octapeptide), matching the
  # cache-dir name configs/configs_data.py expects: {NAME}_lag10ns_from100ns_multiK.
"""

import argparse
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue
from threading import Thread

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ── Bio PKL loading (same as rebuild_msm_emission.py) ─────────────────────────

def _find_bio_path(bio_dir, pdb_id, traj_name, frame_id=None):
    """Locate bio PKL for one frame.

    Naming conventions supported:
      Atlas:   bio/{pdb_id}.pkl.gz   (flat, no subdir)
      BioEmu:  bio/{system}/{traj_name}_{frame_id}.pkl.gz
               where system = traj_name.split("|")[0]

    The frame_id is the integer index in the trajectory, NOT the frame suffix in pdb_id.
    """
    system = traj_name.split("|")[0]
    candidates = [
        # Flat (Atlas-style): bio/{pdb_id}.pkl.gz
        os.path.join(bio_dir, pdb_id + ".pkl.gz"),
        # Subdirectory with traj_name as subdir
        os.path.join(bio_dir, traj_name, pdb_id + ".pkl.gz"),
        # Subdirectory with system (first segment) as subdir, using pdb_id basename
        os.path.join(bio_dir, system, pdb_id + ".pkl.gz"),
    ]
    # BioEmu-style: bio/{system}/{traj_name}_{frame_id}.pkl.gz
    if frame_id is not None:
        candidates.append(os.path.join(bio_dir, system, f"{traj_name}_{frame_id}.pkl.gz"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _try_load_pkl(path):
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def _load_ca_one_frame(bio_dir, pdb_id, traj_name, frame_id=None):
    path = _find_bio_path(bio_dir, pdb_id, traj_name, frame_id=frame_id)
    if path is None:
        return None
    try:
        d = _try_load_pkl(path)
        if "atom_array" in d:
            aa = d["atom_array"]
            mask = aa.atom_name == "CA"
            return aa.coord[mask].astype(np.float32)
        elif "ca_coords" in d:
            return np.asarray(d["ca_coords"], dtype=np.float32)
        return None
    except Exception:
        return None


def _load_ca_threaded(system_rows, bio_dir, n_threads=8):
    """Load all CA coords for a system. Returns {traj: [T,N_ca,3]}."""
    import pandas as pd
    groups = system_rows.groupby("traj_name")
    result = {}

    def worker(q):
        while True:
            item = q.get()
            if item is None:
                break
            traj_name, rows = item
            coords, fids = [], []
            for _, row in rows.iterrows():
                frame_id = int(row["frame_id"]) if "frame_id" in row else None
                ca = _load_ca_one_frame(bio_dir, row["pdb_id"], traj_name, frame_id=frame_id)
                if ca is not None:
                    coords.append(ca)
                    fids.append(int(row["frame_id"]) if frame_id is not None else len(coords) - 1)
            if len(coords) >= 3:
                order = np.argsort(fids)
                result[traj_name] = np.stack([coords[i] for i in order], axis=0)

    q = Queue(maxsize=n_threads * 2)
    threads = [Thread(target=worker, args=(q,)) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for traj_name, rows in groups:
        q.put((traj_name, rows))
    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()
    return result


# ── TICA projection ─────────────────────────────────────────────────────────────

def _project_to_tica(ca_coords, tica_mean, tica_components, idx_i, idx_j):
    """Project [T, N_ca, 3] → [T, N_tica]."""
    diff  = ca_coords[:, idx_i] - ca_coords[:, idx_j]
    dists = np.linalg.norm(diff, axis=-1)
    return (dists - tica_mean) @ tica_components


# ── MSM fitting ─────────────────────────────────────────────────────────────────

def _reversible_count_T(dtrajs, K, lag_frames):
    """Reversible MSM transition matrix at `lag_frames`, estimated from discrete
    trajectories. Symmetrized counts (detailed balance), row-normalized; any
    disconnected state (zero row) gets a self-transition. Returns (T, n_pairs)."""
    C = np.zeros((K, K), dtype=np.float64)
    n_pairs = 0
    for labels in dtrajs:
        labels = np.asarray(labels)
        n = len(labels)
        if n <= lag_frames:
            continue
        s_from, s_to = labels[:-lag_frames], labels[lag_frames:]
        mask = (s_from >= 0) & (s_from < K) & (s_to >= 0) & (s_to < K)
        if not mask.any():
            continue
        n_pairs += int(mask.sum())
        np.add.at(C, (s_from[mask], s_to[mask]), 1.0)
    C_sym = (C + C.T) / 2.0
    row_sums = C_sym.sum(axis=1, keepdims=True)
    T = np.zeros_like(C_sym)
    nz = row_sums.ravel() > 0
    T[nz] = C_sym[nz] / row_sums[nz]
    for i in np.where(~nz)[0]:
        T[i, i] = 1.0
    return T, n_pairs


def _matrix_power_fractional(T, p):
    """T^p for a reversible MSM via the symmetric decomposition S = D^{1/2} T D^{-1/2}
    (D=diag(π)): S=QΛQ^T ⇒ T^p = D^{-1/2}(QΛ^p Q^T)D^{1/2}. Exact and row-stochastic
    for p∈[0,1]. Used to convert a long-lag count matrix to a short-lag operator."""
    K = T.shape[0]
    pi = np.ones(K) / K
    for _ in range(20000):
        pn = pi @ T
        if np.max(np.abs(pn - pi)) < 1e-14:
            pi = pn
            break
        pi = pn
    pi = np.clip(pi, 1e-30, None)
    d_half = np.sqrt(pi)
    d_half_inv = 1.0 / d_half
    S = (d_half[:, None] * T) * d_half_inv[None, :]
    S = 0.5 * (S + S.T)
    evals, Q = np.linalg.eigh(S)
    evals = np.clip(evals, 0.0, 1.0)
    S_p = (Q * np.power(evals, p)[None, :]) @ Q.T
    T_p = (d_half_inv[:, None] * S_p) * d_half[None, :]
    T_p = np.clip(T_p, 0.0, None)
    rs = T_p.sum(axis=1, keepdims=True)
    rs = np.where(rs > 1e-12, rs, 1.0)
    return T_p / rs


def _fit_msm_one_K(tica_list, K, msm_lagtime=2, count_lag_frames=10, root_divisor=10):
    """
    Run k-means with target K, then build a transition matrix over ALL K-means
    states (connectivity threshold=1: any cluster with ≥1 observed frame is valid).

    Unlike MaximumLikelihoodMSM which restricts to the largest connected component
    and labels other frames as -1, we keep every k-means cluster as a valid state.
    This prevents HMM emission NaN caused by frames with no valid state assignment.

    tica_list: list of [T_i, N_tica] arrays (one per traj).
    Returns dict with n_states, centers, state_labels (no -1), T, pi.
    """
    import deeptime

    tica_concat = np.concatenate(tica_list, axis=0)
    total_frames = tica_concat.shape[0]

    actual_k = min(K, max(3, total_frames // 50))
    if actual_k < 3:
        return None

    km = deeptime.clustering.KMeans(n_clusters=actual_k, max_iter=500, init_strategy="kmeans++")
    km.fit(tica_concat)
    km_model = km.fetch_model()
    dtrajs_raw = [km_model.transform(t) for t in tica_list]

    # Transition matrix: reversible count at the long counting lag (count_lag_frames; 10 frames =
    # 100 ns), then matrix-root ^(1/root_divisor) down to the 1-frame (10 ns) effective lag.
    T_count, n_pairs = _reversible_count_T(dtrajs_raw, actual_k, count_lag_frames)
    if n_pairs > 0:
        T = _matrix_power_fractional(T_count, 1.0 / root_divisor).astype(np.float32)
    else:
        # Fallback (trajectories all shorter than the counting lag): short-lag reversible
        # estimate with a pseudocount so T stays row-stochastic.
        min_len = min(len(t) for t in tica_list)
        eff_lag = max(1, min(msm_lagtime, min_len // 10))
        counts_matrix = np.ones((actual_k, actual_k), dtype=np.float64)  # pseudocount=1
        for dtraj in dtrajs_raw:
            dtraj = np.asarray(dtraj)
            for t in range(len(dtraj) - eff_lag):
                counts_matrix[int(dtraj[t]), int(dtraj[t + eff_lag])] += 1.0
        counts_sym = counts_matrix + counts_matrix.T
        T = (counts_sym / counts_sym.sum(axis=1, keepdims=True)).astype(np.float32)

    # Stationary distribution via power iteration (left eigenvector for eigenvalue=1)
    pi = np.ones(actual_k, dtype=np.float64) / actual_k
    for _ in range(2000):
        pi_new = pi @ T.astype(np.float64)
        pi_new /= pi_new.sum()
        if np.max(np.abs(pi_new - pi)) < 1e-12:
            break
        pi = pi_new
    pi = (pi_new / pi_new.sum()).astype(np.float32)

    return {
        "n_states":          actual_k,
        "cluster_centers":   km_model.cluster_centers.astype(np.float32),
        "dtrajs":            dtrajs_raw,   # raw k-means labels, no -1
        "transition_matrix": T,
        "stationary_dist":   pi,
    }


def _build_state_labels(dtrajs, traj_names_ordered):
    """Convert dtrajs list back to {traj: ndarray}."""
    return {tn: np.asarray(dt, dtype=np.int32) for tn, dt in zip(traj_names_ordered, dtrajs)}


# ── GMM fitting ─────────────────────────────────────────────────────────────────

def _fit_gmm_all_C(tica_all, labels_all, n_states, max_C, min_var=1e-4, min_frames=20):
    """Fit GMM for C=1..max_C. Returns {C: {weights, means, precisions_chol}}."""
    from protenix.data.msm_builder import fit_all_gmm_emissions
    return fit_all_gmm_emissions(tica_all, labels_all, n_states, max_C,
                                  min_var=min_var, min_frames_for_gmm=min_frames)


# ── Per-system worker ────────────────────────────────────────────────────────────

def _fit_tica_from_bio(traj_ca, n_tica_dims, lagtime_frames):
    """Fit ONE shared TICA basis per system from all replicas pooled (deeptime list input =
    boundary-aware: lag pairs only within each replica). Mirrors the TICA stage of
    protenix.data.msm_builder.build_msm_for_system, so it reproduces the released caches'
    basis. Returns the basis + per-replica TICA projections, or None if insufficient data.
    """
    import deeptime
    from protenix.data.msm_builder import compute_ca_pairwise_distances

    valid = {tn: c for tn, c in traj_ca.items() if c.shape[0] >= 3}
    if not valid:
        return None
    total_frames = sum(c.shape[0] for c in valid.values())
    actual_nca = next(iter(valid.values())).shape[1]
    # Adaptive max_ca so N_pairs << total_frames (well-conditioned TICA).
    max_ca = min(actual_nca, max(15, int((2 * total_frames / 5) ** 0.5)))

    features_list, names, pair_indices, traj_fc = [], [], None, {}
    for tn, ca in valid.items():
        dists, pair_indices = compute_ca_pairwise_distances(ca, max_ca=max_ca)
        features_list.append(dists.astype(np.float32))
        names.append(tn)
        traj_fc[tn] = ca.shape[0]
    if sum(f.shape[0] for f in features_list) < 50:
        return None

    actual_dims = min(n_tica_dims, features_list[0].shape[1])
    min_len = min(f.shape[0] for f in features_list)
    tica_lag = min(lagtime_frames, max(1, min_len // 5))
    tica_est = deeptime.decomposition.TICA(lagtime=tica_lag, dim=actual_dims)
    try:
        tica_est.fit(features_list)   # one shared basis, all replicas pooled (boundary-aware)
    except Exception as e:
        logger.warning(f"TICA fit failed: {e}")
        return None
    tm = tica_est.fetch_model()
    tica_mean = tm.mean_0.astype(np.float32)
    tica_components = tm.singular_vectors_left[:, :actual_dims].astype(np.float32)
    eig = np.clip(tm.singular_values[:actual_dims], -0.9999, 0.9999).astype(np.float32)
    tica_by_traj = {tn: tm.transform(f).astype(np.float32) for tn, f in zip(names, features_list)}
    return dict(
        tica_mean=tica_mean, tica_components=tica_components, tica_eigenvalues=eig,
        idx_i=pair_indices[0], idx_j=pair_indices[1], n_tica_dims=actual_dims,
        tica_by_traj=tica_by_traj, traj_frame_counts=traj_fc, n_ca_full=actual_nca,
    )


def _process_system(args):
    (
        system_name, system_rows, bio_dir,
        src_gmm_pkl, src_old_pkl,
        out_path, k_values, max_C, msm_lagtime, n_io_threads,
        n_tica_dims_cfg, tica_lagtime_frames,
    ) = args

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ── 1. Load source PKL ──────────────────────────────────────────────────────
    src_pkl = None
    for p in [src_gmm_pkl, src_old_pkl]:
        if p and os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    src_pkl = pickle.load(f)
                break
            except Exception:
                continue
    # ── 2-3. Obtain the TICA basis + per-trajectory TICA projections ─────────────
    tica_by_traj = {}
    if src_pkl is not None:
        # Use a precomputed TICA basis from the source PKL (MSMArtifacts or plain dict).
        if isinstance(src_pkl, dict):
            tica_mean        = src_pkl["tica_mean"]
            tica_components  = src_pkl["tica_components"]
            tica_eigenvalues = src_pkl["tica_eigenvalues"]
            idx_i            = src_pkl["pair_indices_i"]
            idx_j            = src_pkl["pair_indices_j"]
            n_tica_dims      = src_pkl["n_tica_dims"]
            n_ca_full        = src_pkl.get("n_ca_full", 0)
            traj_fc          = src_pkl.get("traj_frame_counts", {})
            cached_tica      = src_pkl.get("tica_coords_by_traj", None)
        else:
            tica_mean        = src_pkl.tica_mean
            tica_components  = src_pkl.tica_components
            tica_eigenvalues = src_pkl.tica_eigenvalues
            idx_i            = src_pkl.pair_indices_i
            idx_j            = src_pkl.pair_indices_j
            n_tica_dims      = src_pkl.n_tica_dims
            n_ca_full        = getattr(src_pkl, "n_ca_full", 0)
            traj_fc          = getattr(src_pkl, "traj_frame_counts", {})
            cached_tica      = getattr(src_pkl, "tica_coords_by_traj", None)

        if cached_tica is not None:
            tica_by_traj = {k: v for k, v in cached_tica.items()}        # fast path
        elif system_rows is None:
            return system_name, "no_cached_tica"
        else:
            traj_ca = _load_ca_threaded(system_rows, bio_dir, n_threads=n_io_threads)
            if not traj_ca:
                return system_name, "no_data"
            for traj_name, ca_arr in traj_ca.items():                   # slow path: project
                if max(idx_i.max(), idx_j.max()) >= ca_arr.shape[1]:
                    continue
                try:
                    tica_by_traj[traj_name] = _project_to_tica(
                        ca_arr, tica_mean, tica_components, idx_i, idx_j
                    ).astype(np.float32)
                except Exception:
                    continue
    else:
        # ── From scratch: fit the TICA basis directly from bioassembly data ──────
        # (self-contained — no precomputed cache required).
        if system_rows is None:
            return system_name, "no_source_and_no_csv"
        traj_ca = _load_ca_threaded(system_rows, bio_dir, n_threads=n_io_threads)
        if not traj_ca:
            return system_name, "no_data"
        fit = _fit_tica_from_bio(traj_ca, n_tica_dims_cfg, tica_lagtime_frames)
        if fit is None:
            return system_name, "tica_fit_failed"
        tica_mean        = fit["tica_mean"]
        tica_components  = fit["tica_components"]
        tica_eigenvalues = fit["tica_eigenvalues"]
        idx_i, idx_j     = fit["idx_i"], fit["idx_j"]
        n_tica_dims      = fit["n_tica_dims"]
        n_ca_full        = fit["n_ca_full"]
        traj_fc          = fit["traj_frame_counts"]
        tica_by_traj     = fit["tica_by_traj"]

    if not tica_by_traj:
        return system_name, "no_tica"

    traj_names_ordered = sorted(tica_by_traj.keys())
    tica_list = [tica_by_traj[tn] for tn in traj_names_ordered]
    tica_all   = np.concatenate(tica_list, axis=0)

    # ── 4. Build MSM + GMM for each K ───────────────────────────────────────────
    by_k = {}

    for K in k_values:
        msm_result = _fit_msm_one_K(tica_list, K, msm_lagtime=msm_lagtime)
        if msm_result is None:
            logger.warning(f"{system_name}: MSM fit failed for K={K}, skipping")
            continue

        n_states  = msm_result["n_states"]
        dtrajs    = msm_result["dtrajs"]
        labels_all = np.concatenate(dtrajs, axis=0).astype(np.int32)

        # Fit GMM for C=1..max_C
        gmm_by_c = _fit_gmm_all_C(tica_all, labels_all, n_states, max_C)

        by_k[K] = {
            "n_states":          n_states,
            "cluster_centers":   msm_result["cluster_centers"],
            "state_labels":      _build_state_labels(dtrajs, traj_names_ordered),
            "transition_matrix": msm_result["transition_matrix"],
            "stationary_dist":   msm_result["stationary_dist"],
            "gmm_by_c":          gmm_by_c,
        }

    if not by_k:
        return system_name, "all_K_failed"

    # ── 5. Save unified dict PKL ─────────────────────────────────────────────────
    out_data = {
        "tica_mean":          tica_mean,
        "tica_components":    tica_components,
        "tica_eigenvalues":   tica_eigenvalues,
        "pair_indices_i":     idx_i,
        "pair_indices_j":     idx_j,
        "n_tica_dims":        n_tica_dims,
        "lagtime_frames":     1,   # T is built as a 1-frame (10 ns) effective-lag operator
                                    # (reversible count @ 100 ns, then ^(1/10) root)
        "n_ca_full":          n_ca_full,
        "traj_frame_counts":  traj_fc,
        "tica_coords_by_traj": {tn: tica_by_traj[tn] for tn in traj_names_ordered},
        "by_k":               by_k,
        "_rebuild_note":      ("multiK MSM; T = reversible count @ 100 ns (count_lag_frames=10) "
                               "rooted ^(1/10) -> 10 ns effective lag; TICA fit from raw bio "
                               "(all replicas pooled, lag 10 ns, 5 dims)"),
    }

    with open(out_path, "wb") as f:
        pickle.dump(out_data, f, protocol=4)

    k_summary = {K: by_k[K]["n_states"] for K in by_k}
    return system_name, f"ok: {k_summary}"


# ── CSV loading helpers ─────────────────────────────────────────────────────────

def _get_system_name(traj_name, dataset):
    """Extract system name from traj name."""
    import re
    if dataset.lower() == "atlas":
        m = re.match(r'^([^_]+_[A-Za-z0-9]+)_R', traj_name)
        return m.group(1) if m else traj_name
    else:
        # BioEmu / CATH2: traj_name is like "cath2_1a87A01|run012_protein"
        return traj_name.split("|")[0]


def _load_csv(csv_dir_or_file, dataset):
    import pandas as pd, pathlib
    p = pathlib.Path(csv_dir_or_file)
    if p.is_file():
        df = pd.read_csv(p)
    else:
        frames = [pd.read_csv(f) for f in sorted(p.glob("*.csv"))]
        df = pd.concat(frames, ignore_index=True)

    # If the CSV already has traj_name and frame_id columns, use them directly.
    # Only parse from pdb_id when these columns are absent (e.g., Atlas CSVs).
    has_traj_col  = "traj_name" in df.columns
    has_frame_col = "frame_id"  in df.columns

    if dataset.lower() == "atlas" and not (has_traj_col and has_frame_col):
        import re
        def parse(pdb_id):
            m = re.match(r'^(.+)_R(\d+)_(\d+)$', pdb_id)
            if m:
                return m.group(1) + "_R" + m.group(2), int(m.group(3))
            m2 = re.match(r'^(.+)_R(\d+)$', pdb_id)
            if m2:
                return m2.group(0), 0
            return pdb_id, 0
        df[["traj_name", "frame_id"]] = pd.DataFrame(
            df["pdb_id"].map(parse).tolist(), index=df.index
        )
    elif not (has_traj_col and has_frame_col):
        # BioEmu CSVs without pre-parsed columns: pdb_id like "cath2_...|run...|frame_042"
        def parse_bioemu(pdb_id):
            parts = pdb_id.split("|")
            traj = "|".join(parts[:2]) if len(parts) >= 2 else pdb_id
            frame = int(parts[2].replace("frame_", "")) if len(parts) >= 3 else 0
            return traj, frame
        df[["traj_name", "frame_id"]] = pd.DataFrame(
            df["pdb_id"].map(parse_bioemu).tolist(), index=df.index
        )
    # else: CSV already has traj_name and frame_id — use as-is

    return df


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset",        default="auto", help="CATH2 / Atlas / auto")
    parser.add_argument("--gmm-cache-dir",  default=None,  help="Rebuilt GMM PKL dir (has tica_coords_by_traj — fast path)")
    parser.add_argument("--cache-dir",      default=None,  help="Original MSM PKL dir (fallback for slow path)")
    parser.add_argument("--from-multiK-dir", default=None, help="Existing multiK PKL dir — iterate directly, no CSV needed")
    parser.add_argument("--csv-dir",        default=None,  help="CSV file or dir with trajectory rows (not needed with --from-multiK-dir)")
    parser.add_argument("--bio-dir",        default=None,  help="Bioassembly PKL dir (for slow-path CA loading)")
    parser.add_argument("--out-dir",        required=True, help="Output dir for new multi-K PKLs")
    parser.add_argument("--k-values",       default="10,20,50", help="Comma-separated K values (default: 10,20,50)")
    parser.add_argument("--max-gmm-components", type=int, default=5, help="Max GMM components C per cluster (default: 5)")
    parser.add_argument("--msm-lagtime",    type=int, default=2,  help="MSM transition counting lagtime (default: 2)")
    parser.add_argument("--n-workers",      type=int, default=32, help="Parallel worker processes")
    parser.add_argument("--n-io-threads",   type=int, default=8,  help="IO threads per worker (slow path)")
    parser.add_argument("--systems",        nargs="*", default=None, help="Subset of system names")
    # From-scratch (no precomputed cache): fit the TICA basis directly from --bio-dir.
    parser.add_argument("--n-tica-dims",       type=int, default=5, help="TICA dimensions when fitting from scratch (default: 5)")
    parser.add_argument("--tica-lagtime-frames", type=int, default=1, help="TICA decomposition lag in frames when fitting from scratch (MSR 10 ns/frame → 1; default: 1)")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", level=logging.INFO,
    )

    k_values = [int(k) for k in args.k_values.split(",")]
    logger.info(f"K values: {k_values},  max GMM C: {args.max_gmm_components}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Fast path: iterate directly over an existing multiK PKL dir (no CSV needed) ──
    if args.from_multiK_dir:
        src_pkls = sorted(f for f in os.listdir(args.from_multiK_dir) if f.endswith("_msm.pkl"))
        if args.systems:
            src_pkls = [f for f in src_pkls if f.replace("_msm.pkl", "") in set(args.systems)]
        logger.info(f"Found {len(src_pkls)} PKLs in {args.from_multiK_dir}")

        worker_args = []
        n_skipped = 0
        for fname in src_pkls:
            system_name = fname.replace("_msm.pkl", "")
            out_path = os.path.join(args.out_dir, fname)
            if os.path.exists(out_path):
                n_skipped += 1
                continue
            src_pkl_path = os.path.join(args.from_multiK_dir, fname)
            worker_args.append((
                system_name,
                None,           # sys_rows — not needed for fast path
                None,           # bio_dir
                src_pkl_path,   # gmm_pkl (has tica_coords_by_traj)
                None,           # old_pkl
                out_path,
                k_values,
                args.max_gmm_components,
                args.msm_lagtime,
                args.n_io_threads,
                args.n_tica_dims,
                args.tica_lagtime_frames,
            ))

        if n_skipped:
            logger.info(f"Skipping {n_skipped} already-built systems")
        logger.info(f"Processing {len(worker_args)} systems with {args.n_workers} workers")
        _run_workers(worker_args, args.n_workers)
        return

    # ── CSV-based path ───────────────────────────────────────────────────────────
    # Two sub-modes:
    #   • with --cache-dir/--gmm-cache-dir : reuse a precomputed TICA basis (original).
    #   • from scratch (no cache dir)       : fit TICA directly from --bio-dir (self-contained).
    from_scratch = (args.cache_dir is None and args.gmm_cache_dir is None)
    if args.csv_dir is None:
        parser.error("--csv-dir is required unless --from-multiK-dir is used")
    if from_scratch and args.bio_dir is None:
        parser.error("--bio-dir is required when building from scratch (no --cache-dir/--gmm-cache-dir)")

    # Load CSV
    logger.info(f"Loading CSV: {args.csv_dir}")
    df = _load_csv(args.csv_dir, args.dataset)
    df["system_name"] = df["traj_name"].apply(lambda t: _get_system_name(t, args.dataset))
    logger.info(f"Loaded {len(df)} rows, {df['system_name'].nunique()} unique systems")

    if from_scratch:
        logger.info("No source cache provided → fitting TICA from scratch from --bio-dir")
    else:
        # Filter to systems with a source PKL available
        src_dir = args.gmm_cache_dir or args.cache_dir
        have_src = set(
            f.replace("_msm.pkl", "") for f in os.listdir(src_dir) if f.endswith("_msm.pkl")
        )
        if args.gmm_cache_dir and args.cache_dir:
            have_src |= set(
                f.replace("_msm.pkl", "") for f in os.listdir(args.cache_dir) if f.endswith("_msm.pkl")
            )
        df = df[df["system_name"].isin(have_src)]
        logger.info(f"Systems with source PKL: {df['system_name'].nunique()}")

    if args.systems:
        df = df[df["system_name"].isin(set(args.systems))]
        logger.info(f"Filtered to {df['system_name'].nunique()} requested systems")

    # Build worker args, skip already done
    worker_args = []
    n_skipped = 0
    for system_name, sys_rows in df.groupby("system_name"):
        out_path = os.path.join(args.out_dir, f"{system_name}_msm.pkl")
        if os.path.exists(out_path):
            n_skipped += 1
            continue
        gmm_pkl = os.path.join(args.gmm_cache_dir, f"{system_name}_msm.pkl") if args.gmm_cache_dir else None
        old_pkl = os.path.join(args.cache_dir, f"{system_name}_msm.pkl") if args.cache_dir else None
        worker_args.append((
            system_name,
            sys_rows.reset_index(drop=True),
            args.bio_dir,
            gmm_pkl,
            old_pkl,
            out_path,
            k_values,
            args.max_gmm_components,
            args.msm_lagtime,
            args.n_io_threads,
            args.n_tica_dims,
            args.tica_lagtime_frames,
        ))

    if n_skipped:
        logger.info(f"Skipping {n_skipped} already-built systems")
    logger.info(f"Processing {len(worker_args)} systems with {args.n_workers} workers")
    _run_workers(worker_args, args.n_workers)


def _run_workers(worker_args, n_workers):
    n_ok, n_fail = 0, 0
    t0 = time.time()
    total = len(worker_args)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_system, a): a[0] for a in worker_args}
        for i, fut in enumerate(as_completed(futures)):
            sname = futures[fut]
            try:
                sname, status = fut.result()
            except Exception as e:
                status = f"exception: {e}"
            if status.startswith("ok"):
                n_ok += 1
            else:
                n_fail += 1
                logger.warning(f"  FAIL {sname}: {status}")
            if (i + 1) % 20 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (total - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  [{i+1}/{total}] ok={n_ok} fail={n_fail} "
                    f"{rate:.1f}/s ETA {eta/60:.0f}min"
                )

    logger.info(f"Done: {n_ok} ok, {n_fail} failed in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
