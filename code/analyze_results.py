import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import load_all, true_wavespeed_np, LX, LZ

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir',  default='results')
    p.add_argument('--data_dir', default='.')
    return p.parse_args()

def load_history(exp_dir):
    path = os.path.join(exp_dir, 'history.npy')
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    return None

EXP_NAMES = ['baseline', 'adaptive_weights', 'activation_sin',
             'activation_adaptive', 'fourier_features',
             'adaptive_colloc', 'best_combined']

LABELS = {
    'baseline'            : 'Baseline (tanh, fixed λ)',
    'adaptive_weights'    : 'Impr. 1: Adaptive λ  [Wang 2021]',
    'activation_sin'      : 'Impr. 2a: Sin (SIREN)  [Sitzmann 2020]',
    'activation_adaptive' : 'Impr. 2b: Adaptive act.  [Jagtap 2020]',
    'fourier_features'    : 'Impr. 3: Fourier features  [Tancik 2020]',
    'adaptive_colloc'     : 'Impr. 4: Adaptive collocation',
    'best_combined'       : 'Best: all improvements',
}

COLOURS = plt.cm.tab10(np.linspace(0, 1, len(EXP_NAMES)))

def available(out_dir):
    return [(n, os.path.join(out_dir, n))
            for n in EXP_NAMES
            if os.path.isdir(os.path.join(out_dir, n))]

def plot_loss_curves(out_dir, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (name, edir), col in zip(available(out_dir), COLOURS):
        h = load_history(edir)
        if not h or not h.get('epoch'): continue
        lbl = LABELS.get(name, name)
        axes[0].semilogy(h['epoch'], h['loss'],       color=col, label=lbl, lw=1.4)
        axes[1].plot    (h['epoch'], h['alpha_rmse'],  color=col, label=lbl, lw=1.4)
    for ax, yl, t in zip(axes,
                          ['Total loss (log)', 'α RMSE (km/s)'],
                          ['Training loss', 'Wavespeed RMSE']):
        ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel(yl, fontsize=11)
        ax.set_title(t, fontsize=12); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'all_loss_curves.png'), dpi=150)
    plt.close(fig)
    print(f"  → all_loss_curves.png")

def plot_lambda_evolution(out_dir, save_dir):
    h = load_history(os.path.join(out_dir, 'adaptive_weights'))
    if not h or not h.get('lam_pde'):
        print("  → lambda_evolution.png  [skipped — no data]"); return
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, col, lbl in [('lam_pde','r','λ_PDE'), ('lam_snap','b','λ_snapshot'),
                         ('lam_bc','k','λ_BC'),   ('lam_obs','c','λ_seismogram')]:
        ax.plot(h['epoch'], h[k], color=col, label=lbl, lw=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('λ')
    ax.set_title('Adaptive loss weights (Wang et al. 2021)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'lambda_evolution.png'), dpi=150)
    plt.close(fig)
    print(f"  → lambda_evolution.png")

def plot_activation_study(out_dir, save_dir):
    act_exps = [('baseline','tanh (baseline)','grey'),
                ('activation_sin','sin [SIREN]','blue'),
                ('activation_adaptive','adaptive tanh','green')]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, lbl, col in act_exps:
        h = load_history(os.path.join(out_dir, name))
        if not h or not h.get('epoch'): continue
        ax.plot(h['epoch'], h['alpha_rmse'], color=col, lw=1.8, label=lbl)
    ax.set_xlabel('Epoch'); ax.set_ylabel('α RMSE (km/s)')
    ax.set_title('Activation function comparison (Improvement 2)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'activation_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"  → activation_comparison.png")

def plot_improvement_overview(out_dir, save_dir):
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    panels = [
        ('Improvement 1: Adaptive Loss Weights',
         [('baseline','grey','--'), ('adaptive_weights','red','-')],
         'α RMSE (km/s)', 'alpha_rmse',
         'Wang et al. 2021 — balances gradient magnitudes'),
        ('Improvement 2: Activation Functions',
         [('baseline','grey','--'),('activation_sin','blue','-'),
          ('activation_adaptive','green','-')],
         'α RMSE (km/s)', 'alpha_rmse',
         'tanh / sin (SIREN) / adaptive — frequency content'),
        ('Improvement 3: Fourier Feature Encoding',
         [('baseline','grey','--'), ('fourier_features','purple','-')],
         'α RMSE (km/s)', 'alpha_rmse',
         'Tancik et al. 2020 — overcomes spectral bias'),
        ('Improvement 4: Adaptive Collocation',
         [('baseline','grey','--'), ('adaptive_colloc','orange','-')],
         'α RMSE (km/s)', 'alpha_rmse',
         'Residual-weighted PDE point resampling'),
    ]

    for ax, (title, exps_list, ylabel, key, desc) in zip(axes, panels):
        for name, col, ls in exps_list:
            h = load_history(os.path.join(out_dir, name))
            if not h or not h.get(key): continue
            ax.plot(h['epoch'], h[key], color=col, ls=ls, lw=1.8,
                    label=LABELS.get(name, name))
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.02, desc, transform=ax.transAxes,
                fontsize=7, color='#555', va='bottom', style='italic')

    fig.suptitle('PINN Improvement Study', fontsize=14)
    fig.savefig(os.path.join(save_dir, 'improvement_overview.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → improvement_overview.png")

def plot_wavespeed_slices(data, out_dir, save_dir):
    xxs, zzs = data['xxs'], data['zzs']
    alpha_true = true_wavespeed_np(xxs * LX, zzs * LZ)
    mid_row = xxs.shape[0] // 2
    mid_col = xxs.shape[1] // 2
    x_vals  = xxs[mid_row, :] * LX
    z_vals  = zzs[:, mid_col] * LZ

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Wavespeed profile slices')
    axes[0].plot(x_vals, alpha_true[mid_row, :], 'k--', lw=2.5, label='True α')
    axes[1].plot(alpha_true[:, mid_col], z_vals,  'k--', lw=2.5, label='True α')

    from matplotlib.lines import Line2D
    legend_handles_h = []
    legend_handles_v = []
    for (name, edir), col in zip(available(out_dir), COLOURS):
        h = load_history(edir)
        if not h or not h.get('alpha_rmse'):
            continue
        final_rmse = h['alpha_rmse'][-1]
        lbl = f"{LABELS.get(name, name)}  (RMSE={final_rmse:.3f})"
        proxy = Line2D([0], [0], color=col, lw=1.8, label=lbl)
        legend_handles_h.append(proxy)
        legend_handles_v.append(proxy)

    for ax, xl, yl, t, handles in zip(
            axes,
            ['x (km)', 'α (km/s)'],
            ['α (km/s)', 'z (km)'],
            [f'Horizontal slice  z={z_vals[mid_row]:.2f} km',
             f'Vertical slice  x={x_vals[mid_col]:.2f} km'],
            [legend_handles_h, legend_handles_v]):
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(t)
        true_handle = Line2D([0], [0], color='k', ls='--', lw=2.5, label='True α')
        ax.legend(handles=[true_handle] + handles, fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'wavespeed_slices.png'), dpi=150)
    plt.close(fig)
    print(f"  → wavespeed_slices.png")

def make_latex_table(out_dir, save_dir):
    csv = os.path.join(out_dir, 'summary', 'summary.csv')
    if not os.path.exists(csv):
        print("  → summary_table.tex  [skipped — no summary.csv]"); return

    tex_labels = {
        'baseline'            : r'Baseline (tanh, fixed $\lambda$)',
        'adaptive_weights'    : r'Impr.~1: Adaptive $\lambda$ \cite{wang2021}',
        'activation_sin'      : r'Impr.~2a: Sin activation \cite{sitzmann2020}',
        'activation_adaptive' : r'Impr.~2b: Adaptive activation \cite{jagtap2020}',
        'fourier_features'    : r'Impr.~3: Fourier features \cite{tancik2020}',
        'adaptive_colloc'     : r'Impr.~4: Adaptive collocation',
        'best_combined'       : r'\textbf{Best: all improvements}',
    }
    rows = [l.strip().split(',') for l in open(csv).readlines()[1:] if l.strip()]
    lines = [
        r'\begin{table}[ht]\centering',
        r'\caption{PINN experiment results, Case~3 (ellipsoidal anomaly).}',
        r'\label{tab:results}',
        r'\begin{tabular}{lcc}\toprule',
        r'Experiment & $\alpha$~RMSE (km/s) & $|\Delta U|$ rel.\ err \\\midrule',
    ]
    for r in rows:
        lbl = tex_labels.get(r[0], r[0])
        lines.append(f'{lbl} & {float(r[1]):.4f} & {float(r[2]):.4f} \\\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']

    out = os.path.join(save_dir, 'summary_table.tex')
    open(out, 'w').write('\n'.join(lines))
    print(f"  → summary_table.tex")

def main():
    args    = parse_args()
    ana_dir = os.path.join(args.out_dir, 'analysis')
    os.makedirs(ana_dir, exist_ok=True)
    data = load_all(data_dir=args.data_dir, device='cpu', verbose=False)

    print(f"[analyze] {os.path.abspath(args.out_dir)}")
    plot_loss_curves         (args.out_dir, ana_dir)
    plot_lambda_evolution    (args.out_dir, ana_dir)
    plot_activation_study    (args.out_dir, ana_dir)
    plot_improvement_overview(args.out_dir, ana_dir)
    plot_wavespeed_slices    (data, args.out_dir, ana_dir)
    make_latex_table         (args.out_dir, ana_dir)

    print(f"\n[analyze] All figures → {os.path.abspath(ana_dir)}/")

if __name__ == '__main__':
    main()