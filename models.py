import numpy as np
import scipy.stats as stats

# ============================================================
# MDS coordinate utilities
# ============================================================

def coords_to_distances(coords):
    """Convert (16, 2) MDS coordinates to (16, 16) distance matrix."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def init_coords_circle(n_items=16):
    """Initialize MDS coordinates on a unit circle."""
    angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, n_items, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)]).flatten()  # (32,)


def get_distances(coords, mds_fixed, n_subjects, n_cond, n_items=16):
    """
    Compute distances array (C, 16, 16) from coords.
    If mds_fixed: same distances for all conditions (unit circle).
    If not mds_fixed: per-subject coords, repeated for each subject's n_cond conditions.
    """
    if mds_fixed:
        fixed_coords = init_coords_circle(n_items).reshape(n_items, 2)
        distances    = coords_to_distances(fixed_coords)                   # (16, 16)
        distances    = distances[None, :, :]                               # (1, 16, 16) — broadcasts
    else:
        # coords: (n_subjects, 16, 2)
        dist_per_subj = np.stack([
            coords_to_distances(coords[s]) for s in range(n_subjects)
        ])                                                                 # (n_subj, 16, 16)
        distances = np.repeat(dist_per_subj, n_cond, axis=0)             # (C, 16, 16)
    return distances


# ============================================================
# SCM
# ============================================================

def unpack_SCM_params(params_vec, n_items=16, n_ss=3,
                      mds_fixed=True, n_subjects=1):
    bias_params = params_vec[:n_items-1]                                   # (15,)
    last_bias   = 1 - bias_params.sum()
    biases      = np.append(bias_params, last_bias)                        # (16,)
    B           = params_vec[n_items-1]                                    # scalar
    c_params    = params_vec[n_items:n_items+n_ss]                        # (3,)

    if mds_fixed:
        coords = None
    else:
        coords = params_vec[n_items+n_ss:].reshape(n_subjects, n_items, 2)  # (n_subj, 16, 2)

    return biases, B, c_params, coords


def SCM(params_vec, prepared_data, mds_fixed=True, n_subjects=1):
    counts, pos_array, ss_idx, _ = prepared_data

    biases, B, c_params, coords = unpack_SCM_params(
        params_vec, mds_fixed=mds_fixed, n_subjects=n_subjects)

    C      = len(ss_idx)
    n_cond = C // n_subjects
    distances = get_distances(coords, mds_fixed, n_subjects, n_cond)      # (1 or C, 16, 16)

    bias_per_condition = np.tile(biases, (C, 1))                           # (C, 16)
    boost_matrix = np.ones_like(bias_per_condition)
    boost_matrix[np.arange(C), pos_array] = B
    bias_per_condition = bias_per_condition * boost_matrix
    bias_per_condition = bias_per_condition / bias_per_condition.sum(axis=1, keepdims=True)

    c_per_condition = c_params[ss_idx][:, None, None]                     # (C, 1, 1)
    sim  = np.exp(-c_per_condition * distances)                           # (C, 16, 16)
    num  = sim * bias_per_condition[:, None, :]
    prob = num / num.sum(axis=2, keepdims=True)

    pred_counts = counts * prob
    return prob, pred_counts


# ============================================================
# SCM-mix
# ============================================================

def unpack_SCM_mix_params(params_vec, n_items=16, n_ss=3,
                           mds_fixed=True, n_subjects=1):
    bias_params   = params_vec[:n_items-1]                                 # (15,)
    last_bias     = 1 - bias_params.sum()
    biases        = np.append(bias_params, last_bias)                      # (16,)
    B             = params_vec[n_items-1]                                  # scalar
    c_params      = params_vec[n_items:n_items+n_ss]                      # (3,)
    memory_params = params_vec[n_items+n_ss:n_items+n_ss*2]               # (3,)
    gamma         = params_vec[n_items+n_ss*2]                             # scalar

    if mds_fixed:
        coords = None
    else:
        coords = params_vec[n_items+n_ss*2+1:].reshape(n_subjects, n_items, 2)

    return biases, B, c_params, memory_params, gamma, coords


def SCM_mix(params_vec, prepared_data, mds_fixed=True, n_subjects=1):
    counts, pos_array, ss_idx, _ = prepared_data

    biases, B, c_params, memory_params, gamma, coords = \
        unpack_SCM_mix_params(params_vec, mds_fixed=mds_fixed, n_subjects=n_subjects)

    C      = len(ss_idx)
    n_cond = C // n_subjects
    distances = get_distances(coords, mds_fixed, n_subjects, n_cond)

    bias_per_condition = np.tile(biases, (C, 1))
    boost_matrix = np.ones_like(bias_per_condition)
    boost_matrix[np.arange(C), pos_array] = B
    bias_per_condition = bias_per_condition * boost_matrix
    bias_per_condition = bias_per_condition / bias_per_condition.sum(axis=1, keepdims=True)

    c_per_condition = c_params[ss_idx][:, None, None]
    sim  = np.exp(-c_per_condition * distances)
    num  = sim * bias_per_condition[:, None, :]
    p_ij = num / num.sum(axis=2, keepdims=True)

    g_j_raw = bias_per_condition ** gamma
    g_j     = g_j_raw / g_j_raw.sum(axis=1, keepdims=True)
    g_j     = g_j[:, None, :]

    memory_per_condition = memory_params[ss_idx][:, None, None]
    prob = memory_per_condition * p_ij + (1 - memory_per_condition) * g_j

    pred_counts = counts * prob
    return prob, pred_counts


# ============================================================
# TCC core
# ============================================================

def compute_pred_probs_TCC(fam_distr_mu_ij, n_points=500, x_range=(-3, 10)):
    squeeze = fam_distr_mu_ij.ndim == 2
    if squeeze:
        fam_distr_mu_ij = fam_distr_mu_ij[None, :, :]

    x_grid = np.linspace(x_range[0], x_range[1], n_points)
    dx     = x_grid[1] - x_grid[0]

    fam_exp = fam_distr_mu_ij[:, :, :, None]
    x_exp   = x_grid[None, None, None, :]

    cdfs            = stats.norm.cdf(x_exp, loc=fam_exp)
    log_cdfs        = np.log(np.clip(cdfs, 1e-300, 1))
    log_prod_all    = log_cdfs.sum(axis=2, keepdims=True)
    log_prod_excl_j = log_prod_all - log_cdfs
    prod_excl_j     = np.exp(log_prod_excl_j)
    pdf_j           = stats.norm.pdf(x_exp, loc=fam_exp)

    pred_probs = np.sum(pdf_j * prod_excl_j * dx, axis=3)

    return pred_probs[0] if squeeze else pred_probs


# ============================================================
# TCC
# ============================================================

def unpack_TCC_params(params_vec, n_items=16, n_ss=3,
                      mds_fixed=True, n_subjects=1):
    d_prime_base = params_vec[0]                                           # scalar
    d_scales     = np.concatenate([[1.0], params_vec[1:n_ss]])            # (3,)
    c            = params_vec[n_ss]                                        # scalar
    biases       = params_vec[n_ss+1:n_ss+1+n_items]                     # (16,)
    hp_boost     = params_vec[n_ss+1+n_items]                             # scalar

    if mds_fixed:
        coords = None
    else:
        coords = params_vec[n_ss+2+n_items:].reshape(n_subjects, n_items, 2)

    return d_prime_base, d_scales, c, biases, hp_boost, coords


def TCC(params_vec, prepared_data, n_points=500, x_range=(-3, 10),
        mds_fixed=True, n_subjects=1):
    counts, pos_array, ss_idx, _ = prepared_data

    d_prime_base, d_scales, c, biases, hp_boost, coords = \
        unpack_TCC_params(params_vec, mds_fixed=mds_fixed, n_subjects=n_subjects)

    C      = len(ss_idx)
    n_cond = C // n_subjects
    distances = get_distances(coords, mds_fixed, n_subjects, n_cond)

    d_scales_per_cond = d_scales[ss_idx]
    d_prime_per_cond  = d_prime_base * d_scales_per_cond

    sim   = np.exp(-c * distances)
    mu_ij = d_prime_per_cond[:, None, None] * sim + biases[None, None, :]
    mu_ij[np.arange(C), :, pos_array] += hp_boost

    pred_probs  = compute_pred_probs_TCC(mu_ij, n_points, x_range)
    pred_counts = counts * pred_probs
    return pred_probs, pred_counts


# ============================================================
# TCC-mix
# ============================================================

def unpack_TCC_mix_params(params_vec, n_items=16, n_ss=3,
                           mds_fixed=True, n_subjects=1):
    d_prime_base = params_vec[0]                                           # scalar
    d_scales     = np.concatenate([[1.0], params_vec[1:n_ss]])            # (3,)
    c            = params_vec[n_ss]                                        # scalar
    biases       = params_vec[n_ss+1:n_ss+1+n_items]                     # (16,)
    hp_boost     = params_vec[n_ss+1+n_items]                             # scalar
    p_mem        = params_vec[n_ss+2+n_items:n_ss+2+n_items+n_ss]        # (3,)
    gamma        = params_vec[n_ss+2+n_items+n_ss]                        # scalar

    if mds_fixed:
        coords = None
    else:
        coords = params_vec[n_ss+3+n_items+n_ss:].reshape(n_subjects, n_items, 2)

    return d_prime_base, d_scales, c, biases, hp_boost, p_mem, gamma, coords


def TCC_mix(params_vec, prepared_data, n_points=500, x_range=(-3, 10),
            mds_fixed=True, n_subjects=1):
    counts, pos_array, ss_idx, _ = prepared_data

    d_prime_base, d_scales, c, biases, hp_boost, p_mem, gamma, coords = \
        unpack_TCC_mix_params(params_vec, mds_fixed=mds_fixed, n_subjects=n_subjects)

    C      = len(ss_idx)
    n_cond = C // n_subjects
    distances = get_distances(coords, mds_fixed, n_subjects, n_cond)

    d_scales_per_cond = d_scales[ss_idx]
    d_prime_per_cond  = d_prime_base * d_scales_per_cond

    sim   = np.exp(-c * distances)
    mu_ij = d_prime_per_cond[:, None, None] * sim + biases[None, None, :]
    mu_ij[np.arange(C), :, pos_array] += hp_boost

    p_tcc = compute_pred_probs_TCC(mu_ij, n_points, x_range)

    biases_boosted = np.tile(biases, (C, 1))
    biases_boosted[np.arange(C), pos_array] += hp_boost
    bias_probs = np.exp(biases_boosted)
    bias_probs = bias_probs / bias_probs.sum(axis=1, keepdims=True)
    g_j_raw    = bias_probs ** gamma
    g_j        = g_j_raw / g_j_raw.sum(axis=1, keepdims=True)
    g_j_exp    = g_j[:, None, :]

    p_mem_per_cond = p_mem[ss_idx][:, None, None]
    pred_probs     = p_mem_per_cond * p_tcc + (1 - p_mem_per_cond) * g_j_exp

    pred_counts = counts * pred_probs
    return pred_probs, pred_counts