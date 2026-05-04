import numpy as np
import scipy.stats as stats

# ============================================================
# Constants (imported from config or passed as args)
# ============================================================
# These are set globally when models are called with prepared_data
# that already encodes N_ITEMS, N_SS via array shapes.

# ============================================================
# SCM
# ============================================================

def unpack_SCM_params(params_vec, n_items=16, n_ss=3):
    n_bias_params = n_items - 1

    bias_params = params_vec[:n_bias_params]                               # (15,)
    B           = params_vec[n_bias_params]                                # scalar
    c_params    = params_vec[n_bias_params+1:n_bias_params+1+n_ss]        # (3,)

    last_bias = 1 - bias_params.sum()
    biases    = np.append(bias_params, last_bias)                          # (16,)

    return biases, B, c_params


def SCM(params_vec, prepared_data):
    distances, counts, pos_array, ss_idx, _ = prepared_data

    biases, B, c_params = unpack_SCM_params(params_vec)

    C = len(ss_idx)

    bias_per_condition = np.tile(biases, (C, 1))                           # (C, 16)
    boost_matrix = np.ones_like(bias_per_condition)
    boost_matrix[np.arange(C), pos_array] = B
    bias_per_condition = bias_per_condition * boost_matrix
    bias_per_condition = bias_per_condition / bias_per_condition.sum(axis=1, keepdims=True)

    c_per_condition = c_params[ss_idx][:, None, None]                     # (C, 1, 1)
    sim  = np.exp(-c_per_condition * distances)                           # (C, 16, 16)
    num  = sim * bias_per_condition[:, None, :]                           # (C, 16, 16)
    prob = num / num.sum(axis=2, keepdims=True)

    pred_counts = counts * prob
    return prob, pred_counts


# ============================================================
# SCM-mix
# ============================================================

def unpack_SCM_mix_params(params_vec, n_items=16, n_ss=3):
    bias_params   = params_vec[:n_items-1]                                 # (15,)
    last_bias     = 1 - bias_params.sum()
    biases        = np.append(bias_params, last_bias)                      # (16,)
    B             = params_vec[n_items-1]                                  # scalar
    c_params      = params_vec[n_items:n_items+n_ss]                      # (3,)
    memory_params = params_vec[n_items+n_ss:n_items+n_ss*2]               # (3,)
    gamma         = params_vec[n_items+n_ss*2]                             # scalar

    return biases, B, c_params, memory_params, gamma


def SCM_mix(params_vec, prepared_data):
    distances, counts, pos_array, ss_idx, _ = prepared_data

    biases, B, c_params, memory_params, gamma = unpack_SCM_mix_params(params_vec)

    C = len(ss_idx)

    bias_per_condition = np.tile(biases, (C, 1))                           # (C, 16)
    boost_matrix = np.ones_like(bias_per_condition)
    boost_matrix[np.arange(C), pos_array] = B
    bias_per_condition = bias_per_condition * boost_matrix
    bias_per_condition = bias_per_condition / bias_per_condition.sum(axis=1, keepdims=True)

    c_per_condition = c_params[ss_idx][:, None, None]                     # (C, 1, 1)
    sim  = np.exp(-c_per_condition * distances)                           # (C, 16, 16)
    num  = sim * bias_per_condition[:, None, :]                           # (C, 16, 16)
    p_ij = num / num.sum(axis=2, keepdims=True)                           # (C, 16, 16)

    g_j_raw = bias_per_condition ** gamma                                  # (C, 16)
    g_j     = g_j_raw / g_j_raw.sum(axis=1, keepdims=True)                # (C, 16)
    g_j     = g_j[:, None, :]                                              # (C, 1, 16)

    memory_per_condition = memory_params[ss_idx][:, None, None]           # (C, 1, 1)
    prob = memory_per_condition * p_ij + (1 - memory_per_condition) * g_j

    pred_counts = counts * prob
    return prob, pred_counts


# ============================================================
# TCC core
# ============================================================

def compute_pred_probs_TCC(fam_distr_mu_ij, n_points=500, x_range=(-3, 10)):
    """
    Compute TCC predicted probabilities from familiarity means.

    Args:
        fam_distr_mu_ij: (C, 16, 16) or (16, 16) array of familiarity means mu_ij
        n_points: number of integration points
        x_range: integration range (x_min, x_max)

    Returns:
        pred_probs: (C, 16, 16) or (16, 16) array of predicted probabilities
    """
    squeeze = fam_distr_mu_ij.ndim == 2
    if squeeze:
        fam_distr_mu_ij = fam_distr_mu_ij[None, :, :]                     # (1, 16, 16)

    x_grid = np.linspace(x_range[0], x_range[1], n_points)
    dx     = x_grid[1] - x_grid[0]

    fam_exp = fam_distr_mu_ij[:, :, :, None]                              # (C, 16, 16, 1)
    x_exp   = x_grid[None, None, None, :]                                 # (1,  1,  1,  n_points)

    cdfs            = stats.norm.cdf(x_exp, loc=fam_exp)                  # (C, 16, 16, n_points)
    log_cdfs        = np.log(np.clip(cdfs, 1e-300, 1))
    log_prod_all    = log_cdfs.sum(axis=2, keepdims=True)                  # (C, 16, 1,  n_points)
    log_prod_excl_j = log_prod_all - log_cdfs                              # (C, 16, 16, n_points)
    prod_excl_j     = np.exp(log_prod_excl_j)
    pdf_j           = stats.norm.pdf(x_exp, loc=fam_exp)

    pred_probs = np.sum(pdf_j * prod_excl_j * dx, axis=3)                # (C, 16, 16)

    return pred_probs[0] if squeeze else pred_probs


# ============================================================
# TCC
# ============================================================

def unpack_TCC_params(params_vec, n_items=16, n_ss=3):
    d_prime_base = params_vec[0]                                           # scalar
    d_scales     = np.concatenate([[1.0], params_vec[1:n_ss]])            # (3,) ss=2 fixed at 1
    c            = params_vec[n_ss]                                        # scalar
    biases       = params_vec[n_ss+1:n_ss+1+n_items]                     # (16,) all free
    hp_boost     = params_vec[n_ss+1+n_items]                             # scalar

    return d_prime_base, d_scales, c, biases, hp_boost


def TCC(params_vec, prepared_data, n_points=500, x_range=(-3, 10)):
    distances, counts, pos_array, ss_idx, _ = prepared_data

    d_prime_base, d_scales, c, biases, hp_boost = unpack_TCC_params(params_vec)

    C = len(ss_idx)

    d_scales_per_cond = d_scales[ss_idx]                                   # (C,)
    d_prime_per_cond  = d_prime_base * d_scales_per_cond                   # (C,)

    sim   = np.exp(-c * distances)                                         # (C, 16, 16)
    mu_ij = d_prime_per_cond[:, None, None] * sim + biases[None, None, :] # (C, 16, 16)
    mu_ij[np.arange(C), :, pos_array] += hp_boost

    pred_probs  = compute_pred_probs_TCC(mu_ij, n_points, x_range)        # (C, 16, 16)
    pred_counts = counts * pred_probs
    return pred_probs, pred_counts


# ============================================================
# TCC-mix
# ============================================================

def unpack_TCC_mix_params(params_vec, n_items=16, n_ss=3):
    d_prime_base = params_vec[0]                                           # scalar
    d_scales     = np.concatenate([[1.0], params_vec[1:n_ss]])            # (3,)
    c            = params_vec[n_ss]                                        # scalar
    biases       = params_vec[n_ss+1:n_ss+1+n_items]                     # (16,)
    hp_boost     = params_vec[n_ss+1+n_items]                             # scalar
    p_mem        = params_vec[n_ss+2+n_items:n_ss+2+n_items+n_ss]        # (3,)
    gamma        = params_vec[n_ss+2+n_items+n_ss]                        # scalar

    return d_prime_base, d_scales, c, biases, hp_boost, p_mem, gamma


def TCC_mix(params_vec, prepared_data, n_points=500, x_range=(-3, 10)):
    distances, counts, pos_array, ss_idx, _ = prepared_data

    d_prime_base, d_scales, c, biases, hp_boost, p_mem, gamma = \
        unpack_TCC_mix_params(params_vec)

    C = len(ss_idx)

    d_scales_per_cond = d_scales[ss_idx]
    d_prime_per_cond  = d_prime_base * d_scales_per_cond

    sim   = np.exp(-c * distances)                                         # (C, 16, 16)
    mu_ij = d_prime_per_cond[:, None, None] * sim + biases[None, None, :] # (C, 16, 16)
    mu_ij[np.arange(C), :, pos_array] += hp_boost

    p_tcc = compute_pred_probs_TCC(mu_ij, n_points, x_range)              # (C, 16, 16)

    # guessing state: softmax(biases + hp_boost for HP item)^gamma
    biases_boosted = np.tile(biases, (C, 1))                               # (C, 16)
    biases_boosted[np.arange(C), pos_array] += hp_boost
    bias_probs = np.exp(biases_boosted)
    bias_probs = bias_probs / bias_probs.sum(axis=1, keepdims=True)
    g_j_raw    = bias_probs ** gamma
    g_j        = g_j_raw / g_j_raw.sum(axis=1, keepdims=True)
    g_j_exp    = g_j[:, None, :]                                           # (C, 1, 16)

    p_mem_per_cond    = p_mem[ss_idx][:, None, None]                      # (C, 1, 1)
    pred_probs        = p_mem_per_cond * p_tcc + (1 - p_mem_per_cond) * g_j_exp

    pred_counts = counts * pred_probs
    return pred_probs, pred_counts
