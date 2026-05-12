import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

from models import (
    SCM, SCM_mix, TCC, TCC_mix,
    unpack_SCM_params, unpack_SCM_mix_params,
    unpack_TCC_params, unpack_TCC_mix_params,
)

# ============================================================
# Item colors (Lab color space, matching experiment stimuli)
# ============================================================

def make_item_colors(L=50, a_center=10, b_center=10, radius=40, n_items=16):
    angles     = np.linspace(0, -360, n_items, endpoint=False)
    angles_rad = np.deg2rad(angles)
    colors = []
    for angle in angles_rad:
        a   = a_center + radius * np.cos(angle)
        b   = b_center + radius * np.sin(angle)
        lab = LabColor(L, a, b)
        rgb = convert_color(lab, sRGBColor)
        r   = np.clip(rgb.clamped_rgb_r, 0, 1)
        g   = np.clip(rgb.clamped_rgb_g, 0, 1)
        b_  = np.clip(rgb.clamped_rgb_b, 0, 1)
        colors.append((r, g, b_))
    return colors

item_colors = make_item_colors()


# ============================================================
# Data loading
# ============================================================

def parse_subject_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    blocks   = re.split(r'\n{2,}', content.strip())
    matrices = []
    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        rows = []
        for line in lines:
            values = list(map(int, line.split()))
            rows.append(values[1:])
        mat = np.array(rows)
        if mat.shape == (16, 16):
            matrices.append(mat)
    return matrices


def load_data(data_folder, set_sizes, positions):
    data = {}
    for filepath in sorted(glob.glob(f'{data_folder}/coloridentification_s*.txt')):
        subject_id = int(re.search(r's(\d+)\.txt', filepath).group(1))
        matrices   = parse_subject_file(filepath)

        individual = {}
        for j, ss in enumerate(set_sizes):
            individual[ss] = {}
            for i, pos in enumerate(positions):
                idx = j * len(positions) + i
                individual[ss][pos] = {'data': matrices[idx]}

        aggregated = {}
        for j, ss in enumerate(set_sizes):
            aggregated[ss] = matrices[12 + j]

        data[subject_id] = {'individual': individual, 'aggregated': aggregated}
    return data


def attach_mds_distances(data, subjects, set_sizes, positions,
                         use_single=True, mds_folder='MDS_results'):
    """Attach MDS coordinates and compute pairwise Euclidean distances."""
    n_items = 16
    if use_single:
        r      = 1
        angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, n_items, endpoint=False)
        coords = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
        for subj in subjects:
            for ss in set_sizes:
                for pos in positions:
                    data[subj]['individual'][ss][pos]['mds_coords'] = coords
    else:
        mds_solution = pd.read_csv(f'{mds_folder}/stimulus_coordinates.csv')
        mds_weights  = pd.read_csv(f'{mds_folder}/subject_weights.csv')
        base_coords  = mds_solution.loc[:, 'dim1':'dim2'].values
        for subj in subjects:
            for ss in set_sizes:
                for pos in positions:
                    w = mds_weights.loc[
                        (mds_weights['subject'] == subj) &
                        (mds_weights['set_size'] == ss) &
                        (mds_weights['position'] == pos),
                        'dim1':'dim2'
                    ].values
                    data[subj]['individual'][ss][pos]['mds_coords'] = base_coords * w

    for subj in subjects:
        for ss in set_sizes:
            for pos in positions:
                coords    = data[subj]['individual'][ss][pos]['mds_coords']
                distances = np.linalg.norm(
                    coords[:, None, :] - coords[None, :, :], axis=-1
                )
                data[subj]['individual'][ss][pos]['mds_distances'] = distances

    return data


# ============================================================
# Data preparation
# ============================================================

def extract_observed(data, subjects, set_sizes, positions):
    """Return observed count matrix (C, 16, 16)."""
    observed = []
    for subj in subjects:
        for ss in set_sizes:
            for pos in positions:
                observed.append(data[subj]['individual'][ss][pos]['data'])
    return np.array(observed)


def prepare_data(data, subjects, set_sizes, positions):
    """
    Prepare data arrays for model fitting.

    Returns:
        distances:      (C, 16, 16)
        counts:         (C, 16, 1)
        pos_array:      (C,)  0-indexed HP position
        ss_idx:         (C,)  0-indexed set size
        all_conditions: list of (subj, ss, pos) tuples
    """
    all_conditions = []
    distances      = []
    counts         = []
    pos_array      = []
    ss_idx         = []

    for subj in subjects:
        for i_ss, ss in enumerate(set_sizes):
            for pos in positions:
                all_conditions.append((subj, ss, pos))
                distances.append(data[subj]['individual'][ss][pos]['mds_distances'])
                counts.append(data[subj]['individual'][ss][pos]['data'].sum(axis=1, keepdims=True))
                pos_array.append(pos - 1)
                ss_idx.append(i_ss)

    return (
        np.array(distances),   # (C, 16, 16)
        np.array(counts),      # (C, 16, 1)
        np.array(pos_array),   # (C,)
        np.array(ss_idx),      # (C,)
        all_conditions,
    )


# ============================================================
# Optimization
# ============================================================

def validity_checks(params, model):
    if model is SCM:
        biases, _, _ = unpack_SCM_params(params)
        if np.any(biases <= 0):
            return False

    elif model is SCM_mix:
        biases, _, _, memory_params, gamma = unpack_SCM_mix_params(params)
        if (np.any(biases <= 0) or
                np.any(memory_params < 0) or
                np.any(memory_params > 1) or
                gamma < 0):
            return False

    elif model is TCC:
        return True

    elif model is TCC_mix:
        _, _, _, _, hp_boost, p_mem, gamma = unpack_TCC_mix_params(params)
        if np.any(p_mem < 0) or np.any(p_mem > 1) or gamma < 0 or hp_boost < 0:
            return False

    else:
        return False

    return True


def fit_model(loss_func, initial_params, params_bounds, model,
              observed_counts, prep_data,
              n_points=500, x_range=(-3, 10),
              warmup_options=None, final_options=None, print_every=10):

    if warmup_options is None:
        warmup_options = {'ftol': 1e-9, 'gtol': 1e-3, 'maxiter': 1000,  'maxfun': 10000}
    if final_options is None:
        final_options  = {'ftol': 1e-9, 'gtol': 1e-3, 'maxiter': 10000, 'maxfun': 100000}

    def make_callback(stage):
        iterations = [0]
        def callback(x):
            iterations[0] += 1
            if iterations[0] % print_every == 0:
                if model is TCC or model is TCC_mix:
                    loss = loss_func(x, model, observed_counts, prep_data,
                                     n_points=n_points, x_range=x_range)
                else:
                    loss = loss_func(x, model, observed_counts, prep_data)
                print(f"  [{stage}] Iteration {iterations[0]:4d}: loss = {loss:.6f}")
        return callback

    if model is SCM or model is SCM_mix:
        args_tuple = (model, observed_counts, prep_data)
    elif model is TCC or model is TCC_mix:
        args_tuple = (model, observed_counts, prep_data, n_points, x_range)

    print("=== Warmup ===")
    fit_warmup = minimize(
        loss_func, initial_params,
        args=args_tuple,
        bounds=params_bounds, method='L-BFGS-B',
        callback=make_callback('warmup'),
        options=warmup_options
    )
    print(f"Warmup done: {fit_warmup.message} | Loss: {fit_warmup.fun:.6f}\n")

    print("=== Final fit ===")
    fit_final = minimize(
        loss_func, fit_warmup.x,
        args=args_tuple,
        bounds=params_bounds, method='L-BFGS-B',
        callback=make_callback('final'),
        options=final_options
    )
    print(f"Final done:  {fit_final.message} | Loss: {fit_final.fun:.6f}")

    return fit_final


# ============================================================
# Model comparison metrics
# ============================================================

def compute_aic_bic(fit, total_counts):
    k   = len(fit.x)
    nll = fit.fun * total_counts
    aic = 2 * k + 2 * nll
    bic = k * np.log(total_counts) + 2 * nll
    return k, nll, aic, bic


def print_model_comparison(models_fit, total_counts):
    k_values   = {}
    nll_values = {}
    aic_values = {}
    bic_values = {}

    for name, fit in models_fit.items():
        k, nll, aic, bic = compute_aic_bic(fit, total_counts)
        k_values[name]   = k
        nll_values[name] = nll
        aic_values[name] = aic
        bic_values[name] = bic

    best_aic = min(aic_values, key=aic_values.get)
    best_bic = min(bic_values, key=bic_values.get)

    print(f"{'Model':<22} {'n_params':<13} {'NLL':<13} {'AIC':<8} {'AIC_best':<14} {'BIC':<8} {'BIC_best'}")
    print("-" * 95)
    for name, fit in models_fit.items():
        aic_flag = '***' if name == best_aic else ''
        bic_flag = '***' if name == best_bic else ''
        print(f"{name:<24} {k_values[name]:<10} {nll_values[name]:<12.2f} "
              f"{aic_values[name]:<12.2f} {aic_flag:<10} "
              f"{bic_values[name]:<12.2f} {bic_flag:<10}")

    return pd.DataFrame({
        'model': list(models_fit.keys()),
        'K':     list(k_values.values()),
        'NLL':   list(nll_values.values()),
        'AIC':   list(aic_values.values()),
        'BIC':   list(bic_values.values()),
    })


# ============================================================
# Correlations
# ============================================================

def compute_correlations(pred_probs, observed_counts,
                         set_sizes, positions, subjects, n_ss, n_pos):
    n_subj = len(list(subjects))
    n_cond = n_ss * n_pos

    obs_counts = observed_counts.reshape(n_subj, n_cond, 16, 16)
    obs_props  = obs_counts / (obs_counts.sum(axis=-1, keepdims=True) + 1e-12)
    obs_mean   = obs_props.mean(axis=0)

    pred_mean  = pred_probs.reshape(n_subj, n_cond, 16, 16).mean(axis=0)

    print(f"{'Condition':<20} {'r':<10} {'p-value':<15}")
    print("-" * 45)

    results = []
    for i, ss in enumerate(set_sizes):
        for j, pos in enumerate(positions):
            cond_idx = i * n_pos + j
            r, p     = pearsonr(pred_mean[cond_idx].flatten(),
                                obs_mean[cond_idx].flatten())
            results.append({'set_size': ss, 'position': pos, 'r': r, 'p': p})
            print(f"SS={ss}, Pos={pos:<8} r={r:.3f}    p={p:.3f}")

    return pd.DataFrame(results)


# ============================================================
# Visualizations
# ============================================================

def plot_predicted_vs_observed(pred_probs, observed_counts,
                               set_sizes, positions, subjects,
                               n_ss, n_pos,
                               model_name='Model',
                               error_type='ci',
                               markersize=6,
                               legend_fontsize=10,
                               legend_title_fontsize=15,
                               legend_markersize=15,
                               save_path=None):
    n_subj = len(list(subjects))
    n_cond = n_ss * n_pos

    obs_counts    = observed_counts.reshape(n_subj, n_cond, 16, 16)
    obs_props     = obs_counts / (obs_counts.sum(axis=-1, keepdims=True) + 1e-12)
    pred_reshaped = pred_probs.reshape(n_subj, n_cond, 16, 16)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=item_colors[row],
                   markersize=legend_markersize,
                   label=f'Item {row+1}')
        for row in range(16)
    ]

    fig = plt.figure(figsize=(14, 11))

    legend_ax = fig.add_axes([0, 0.88, 1, 0.06])
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_elements, loc='center', ncol=8,
                     fontsize=legend_fontsize,
                     title='Studied item',
                     title_fontsize=legend_title_fontsize,
                     frameon=True, labelspacing=0.85)

    fig.suptitle(f'{model_name}: Predicted vs Observed\n(all subjects)',
                 fontsize=20, fontweight='bold', y=1.01)

    gs   = fig.add_gridspec(n_ss, n_pos, top=0.84, bottom=0.08,
                             left=0.08, right=0.98, hspace=0.15, wspace=0.2)
    axes = gs.subplots(sharex=True, sharey=True)

    for i, ss in enumerate(set_sizes):
        for j, pos in enumerate(positions):
            ax       = axes[i, j]
            cond_idx = i * n_pos + j

            pred_cond = pred_reshaped[:, cond_idx, :, :]
            obs_cond  = obs_props[:, cond_idx, :, :]

            for row in range(16):
                pred_mean = pred_cond[:, row, :].mean(axis=0)
                pred_std  = pred_cond[:, row, :].std(axis=0)
                obs_mean  = obs_cond[:, row, :].mean(axis=0)
                obs_std   = obs_cond[:, row, :].std(axis=0)

                if error_type == 'ci':
                    t_crit = stats.t.ppf(0.975, df=n_subj - 1)
                    xerr   = t_crit * pred_std / np.sqrt(n_subj)
                    yerr   = t_crit * obs_std  / np.sqrt(n_subj)
                else:
                    xerr = pred_std / np.sqrt(n_subj)
                    yerr = obs_std  / np.sqrt(n_subj)

                ax.errorbar(pred_mean, obs_mean, yerr=yerr, xerr=xerr,
                            fmt='o', markersize=markersize, alpha=0.7,
                            elinewidth=0.8, capsize=2, color=item_colors[row])

            ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'SS={ss}, Pos={pos}', fontsize=10)

    y_label = ('Observed Probability\n(mean ± 95% CI)' if error_type == 'ci'
               else 'Observed Probability\n(mean ± SEM)')
    x_label = ('Predicted Probability\n(mean ± 95% CI)' if error_type == 'ci'
               else 'Predicted Probability\n(mean ± SEM)')

    fig.text(0.5,  0.02, x_label, ha='center', fontsize=15, fontweight='bold')
    fig.text(0.01, 0.45, y_label, va='center', rotation='vertical',
             multialignment='center', fontsize=15, fontweight='bold')
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f'{model_name}.png'),
                    dpi=150, bbox_inches='tight')
        print(f'Saved: {os.path.join(save_path, model_name)}.png')
    
    plt.show()


def plot_predicted_vs_observed_comparison(predictions_dict, observed_counts,
                                          subjects,
                                          n_ss, n_pos,
                                          layout=None,
                                          error_type='sem',
                                          markersize=8,
                                          legend_fontsize=12,
                                          legend_title_fontsize=13,
                                          legend_markersize=12,
                                          save_path=None):
    n_subj = len(list(subjects))
    n_cond = n_ss * n_pos

    obs_counts = observed_counts.reshape(n_subj, n_cond, 16, 16)
    obs_props  = obs_counts / (obs_counts.sum(axis=-1, keepdims=True) + 1e-12)

    if layout is None:
        layout = [['SCM', 'SCM-mix'], ['TCC', 'TCC-mix']]

    nrows = len(layout)
    ncols = max(len(row) for row in layout)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=item_colors[row],
                   markersize=legend_markersize,
                   label=f'Item {row+1}')
        for row in range(16)
    ]

    fig = plt.figure(figsize=(6 * ncols, 6 * nrows))

    legend_ax = fig.add_axes([0, 0.88, 1, 0.06])
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_elements, loc='center', ncol=8,
                     fontsize=legend_fontsize,
                     title='Studied item',
                     title_fontsize=legend_title_fontsize,
                     frameon=True, labelspacing=0.5)

    fig.suptitle('Predicted vs Observed\n(averaged across Set-Size & Position)',
                 fontsize=legend_title_fontsize + 4, fontweight='bold', y=1.0)

    gs   = fig.add_gridspec(nrows, ncols, top=0.84, bottom=0.1,
                          left=0.1, right=0.98,
                          hspace=0.15, wspace=0.05)
    axes = gs.subplots(sharex=True, sharey=True)
    if nrows == 1:
        axes = axes[None, :]

    for row_idx, row_layout in enumerate(layout):
        for col_idx, model_name in enumerate(row_layout):
            ax = axes[row_idx, col_idx]

            if model_name is None or model_name not in predictions_dict:
                ax.set_visible(False)
                continue

            pred_probs    = predictions_dict[model_name][0]
            pred_reshaped = pred_probs.reshape(n_subj, n_cond, 16, 16)
            n             = n_subj * n_cond

            for item in range(16):
                pred_item = pred_reshaped[:, :, item, :]
                obs_item  = obs_props[:, :, item, :]

                pred_mean = pred_item.mean(axis=(0, 1))
                pred_std  = pred_item.std(axis=(0, 1))
                obs_mean  = obs_item.mean(axis=(0, 1))
                obs_std   = obs_item.std(axis=(0, 1))

                if error_type == 'ci':
                    t_crit = stats.t.ppf(0.975, df=n - 1)
                    xerr   = t_crit * pred_std / np.sqrt(n)
                    yerr   = t_crit * obs_std  / np.sqrt(n)
                else:
                    xerr = pred_std / np.sqrt(n)
                    yerr = obs_std  / np.sqrt(n)

                ax.errorbar(pred_mean, obs_mean, yerr=yerr, xerr=xerr,
                            fmt='o', markersize=markersize, alpha=0.7,
                            elinewidth=0.8, capsize=2, color=item_colors[item])

            ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
            ax.set_title(model_name, fontsize=legend_title_fontsize, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    y_label = ('Observed Probability\n(mean ± 95% CI)' if error_type == 'ci'
               else 'Observed Probability\n(mean ± SEM)')
    x_label = ('Predicted Probability\n(mean ± 95% CI)' if error_type == 'ci'
               else 'Predicted Probability\n(mean ± SEM)')

    fig.text(0.5,  0.02, x_label, ha='center',
             fontsize=legend_title_fontsize + 1, fontweight='bold')
    fig.text(0.02, 0.5,  y_label, va='center', rotation='vertical',
             multialignment='center',
             fontsize=legend_title_fontsize + 1, fontweight='bold')
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f'all_models_pred_vs_obs.png'),
                    dpi=150, bbox_inches='tight')
        print(f'Saved: {os.path.join(save_path, "all_models_pred_vs_obs.png")}')
    
    plt.show()


def plot_aic_bic(comparison_df, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    x     = np.arange(len(comparison_df['model']))
    width = 0.35

    ax.bar(x - 0.55 * width, comparison_df['AIC'], width, color='blue', label='AIC')
    ax.bar(x + 0.55 * width, comparison_df['BIC'], width, color='red',  label='BIC')

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['model'], fontsize=13, fontweight='bold')
    miny = comparison_df[['AIC', 'BIC']].min().min() * 0.97
    maxy = comparison_df[['AIC', 'BIC']].max().max() * 1.02
    ax.set_ylim([miny, maxy])
    ax.legend(fontsize=15, loc='upper right', ncol=2)
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, 'AIC_BIC.png'),
                    dpi=150, bbox_inches='tight')
        print(f'Saved: {os.path.join(save_path, "AIC_BIC.png")}')
    
    plt.show()