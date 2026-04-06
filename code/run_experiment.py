import os, sys, argparse, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_all, plot_inputs
from train_pinn  import PINNTrainer, TrainConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device',   default='cuda',
                   help='cuda | cpu  (default: cuda)')
    p.add_argument('--quick',    action='store_true',
                   help='500-epoch smoke run')
    p.add_argument('--epochs',   type=int, default=30_000)
    p.add_argument('--out_dir',  default='results')
    p.add_argument('--data_dir', default='.')
    return p.parse_args()


def make_experiments(args, n_epochs, device):
    base = dict(
        n_epochs             = n_epochs,
        batch_pde            = 10_000,
        lr                   = 1e-3,
        lr_decay             = True,
        hidden_layers        = 4,
        neurons              = 50,
        print_every          = max(1, n_epochs // 30),
        save_every           = max(1, n_epochs // 5),
        lam_pde              = 0.1,
        lam_snap             = 1.0,
        lam_bc               = 0.1,
        lam_obs              = 1.0,
        use_adaptive_weights = False,
        use_adaptive_colloc  = False,
        use_fourier          = False,
        activation           = 'tanh',
        device               = device,
        out_dir              = args.out_dir,
    )

    exps = []

    exps.append(TrainConfig(**{**base,
        'name'   : 'baseline',
        'out_dir': os.path.join(args.out_dir, 'baseline'),
    }))

    exps.append(TrainConfig(**{**base,
        'name'                : 'adaptive_weights',
        'use_adaptive_weights': True,
        'weight_update_every' : max(200, n_epochs // 100),
        'out_dir'             : os.path.join(args.out_dir, 'adaptive_weights'),
    }))

    exps.append(TrainConfig(**{**base,
        'name'      : 'activation_sin',
        'activation': 'sin',
        'lr'        : 5e-4,
        'out_dir'   : os.path.join(args.out_dir, 'activation_sin'),
    }))

    exps.append(TrainConfig(**{**base,
        'name'      : 'activation_adaptive',
        'activation': 'adaptive',
        'out_dir'   : os.path.join(args.out_dir, 'activation_adaptive'),
    }))

    exps.append(TrainConfig(**{**base,
        'name'         : 'fourier_features',
        'use_fourier'  : True,
        'fourier_m'    : 64,
        'fourier_sigma': 5.0,
        'out_dir'      : os.path.join(args.out_dir, 'fourier_features'),
    }))

    exps.append(TrainConfig(**{**base,
        'name'               : 'adaptive_colloc',
        'use_adaptive_colloc': True,
        'resample_every'     : max(200, n_epochs // 100),
        'out_dir'            : os.path.join(args.out_dir, 'adaptive_colloc'),
    }))

    exps.append(TrainConfig(**{**base,
        'name'                : 'best_combined',
        'use_adaptive_weights': True,
        'use_adaptive_colloc' : True,
        'use_fourier'         : True,
        'fourier_m'           : 64,
        'fourier_sigma'       : 5.0,
        'weight_update_every' : max(200, n_epochs // 100),
        'resample_every'      : max(200, n_epochs // 100),
        'out_dir'             : os.path.join(args.out_dir, 'best_combined'),
    }))

    return exps


def run_one(cfg, data):
    os.makedirs(cfg.out_dir, exist_ok=True)
    trainer = PINNTrainer(cfg, data)
    t0 = time.time()
    history = trainer.train()
    elapsed = time.time() - t0
    ev = trainer.evaluate()
    print(f"  → α-RMSE={ev['alpha_rmse']:.4f} km/s  "
          f"U-err={ev['wavefield_rel_err']:.4f}  "
          f"time={elapsed:.0f}s")
    return dict(name=cfg.name, history=history,
                alpha_rmse=ev['alpha_rmse'],
                wavefield_err=ev['wavefield_rel_err'],
                elapsed=elapsed,
                alpha_pred=ev['alpha_pred'],
                alpha_true=ev['alpha_true'],
                U_pred=ev['U_pred'], U_true=ev['U_true'])


def make_summary(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "═"*68)
    print(f"  {'Experiment':<25}  {'α-RMSE':>10}  {'|ΔU| rel':>10}  {'time (s)':>9}")
    print("─"*68)
    for r in results:
        print(f"  {r['name']:<25}  {r['alpha_rmse']:>10.4f}  "
              f"{r['wavefield_err']:>10.4f}  {r['elapsed']:>9.0f}")
    print("═"*68)

    with open(os.path.join(out_dir, 'summary.csv'), 'w') as f:
        f.write('name,alpha_rmse,wavefield_rel_err,elapsed_s\n')
        for r in results:
            f.write(f"{r['name']},{r['alpha_rmse']:.6f},"
                    f"{r['wavefield_err']:.6f},{r['elapsed']:.1f}\n")

    n  = len(results)
    cols = plt.cm.tab10(np.linspace(0, 1, n))
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.5*n))
    if n == 1: axes = axes[None, :]
    fig.suptitle('Wavespeed recovery — all experiments', fontsize=12)
    for i, r in enumerate(results):
        at = r['alpha_true']; ap = r['alpha_pred']
        vmin, vmax = at.min(), at.max()
        axes[i,0].contourf(at, 50, cmap='jet', vmin=vmin, vmax=vmax)
        axes[i,1].contourf(ap, 50, cmap='jet', vmin=vmin, vmax=vmax)
        axes[i,2].contourf(np.abs(at - ap), 50, cmap='hot')
        axes[i,0].set_title(f'{r["name"]} — True',  fontsize=7)
        axes[i,1].set_title(f'Predicted  RMSE={r["alpha_rmse"]:.3f}', fontsize=7)
        axes[i,2].set_title('|Δα|', fontsize=7)
        for ax in axes[i]: ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'wavespeed_comparison.png'), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r, col in zip(results, cols):
        h = r['history']
        if not h['epoch']: continue
        axes[0].semilogy(h['epoch'], h['loss'],      color=col, label=r['name'])
        axes[1].plot    (h['epoch'], h['alpha_rmse'], color=col, label=r['name'])
    for ax, yl, t in zip(axes,
                          ['Total loss','α RMSE (km/s)'],
                          ['Training loss','Wavespeed RMSE']):
        ax.set_xlabel('Epoch'); ax.set_ylabel(yl); ax.set_title(t)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'loss_comparison.png'), dpi=130)
    plt.close(fig)

    names = [r['name'] for r in results]
    rmses = [r['alpha_rmse'] for r in results]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, rmses, color=cols)
    ax.set_ylabel('α RMSE (km/s)')
    ax.set_title('Final wavespeed RMSE by experiment')
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'rmse_bar_chart.png'), dpi=130)
    plt.close(fig)

    print(f"\n[summary] → {out_dir}/")


def main():
    args = parse_args()
    n_epochs = 500 if args.quick else args.epochs

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available — using CPU.")
        device = 'cpu'
    else:
        device = args.device

    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"[GPU] {props.name}  {props.total_memory/1e9:.1f} GB VRAM")

    print("="*65)
    print(f"  AMATH 445 — PINNs for Seismic Inversion  (PyTorch)")
    print(f"  device={device}  epochs={n_epochs}  out={args.out_dir}")
    print("="*65)

    data = load_all(data_dir=args.data_dir, device=device, verbose=True)
    plot_inputs(data, save_dir=os.path.join(args.out_dir, 'data_diagnostics'))

    exps = make_experiments(args, n_epochs, device)
    all_results = []
    for cfg in exps:
        print(f"\n{'━'*65}")
        print(f"  EXPERIMENT: {cfg.name}")
        print(f"{'━'*65}")
        all_results.append(run_one(cfg, data))

    make_summary(all_results, os.path.join(args.out_dir, 'summary'))
    print("\n✓ All experiments complete.")


if __name__ == '__main__':
    main()
