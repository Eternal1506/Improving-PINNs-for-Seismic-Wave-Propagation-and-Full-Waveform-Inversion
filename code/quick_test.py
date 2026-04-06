import sys, os, argparse
import torch

PASS = '✓'
FAIL = '✗'

def section(msg):
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print('─'*60)

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
if device == 'cuda' and not torch.cuda.is_available():
    print("[WARNING] CUDA not available — running on CPU.")
    device = 'cpu'


section("1. Imports & device")
try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from pinn_core   import MLP, FourierEncoding, SinActivation, AdaptiveActivation
    from data_loader import load_all, true_wavespeed_np
    from train_pinn  import PINNModel, TrainConfig, PINNTrainer, compute_losses
    print(f"  {PASS} All imports OK")
    print(f"  {PASS} PyTorch {torch.__version__}")
    if device == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"  {PASS} GPU: {props.name}  {props.total_memory/1e9:.1f} GB")
    else:
        print(f"  {PASS} Running on CPU")
except Exception as e:
    print(f"  {FAIL} {e}"); import traceback; traceback.print_exc(); sys.exit(1)


section("2. pinn_core")
try:
    import torch

    ub  = torch.ones(1, 3)
    net = MLP([3, 20, 20, 1], activation='tanh', ub=ub).to(device)
    X   = torch.rand(50, 3, device=device)
    y   = net(X)
    assert y.shape == (50, 1), y.shape
    print(f"  {PASS} MLP tanh  output={tuple(y.shape)}")

    X.requires_grad_(True)
    y = net(X)
    g = torch.autograd.grad(y.sum(), X, create_graph=True)[0]
    assert g.shape == (50, 3)
    print(f"  {PASS} Autograd through MLP  grad={tuple(g.shape)}")

    ff  = FourierEncoding(3, 16, 5.0).to(device)
    enc = ff(X)
    assert enc.shape == (50, 32)
    print(f"  {PASS} FourierEncoding  enc={tuple(enc.shape)}")

    net2 = MLP([3, 20, 1], activation='adaptive', ub=ub).to(device)
    y2   = net2(X)
    print(f"  {PASS} Adaptive activation  output={tuple(y2.shape)}")

    net3 = MLP([3, 20, 1], activation='sin', ub=ub).to(device)
    y3   = net3(X)
    print(f"  {PASS} Sin activation  output={tuple(y3.shape)}")

except Exception as e:
    print(f"  {FAIL} {e}"); import traceback; traceback.print_exc(); sys.exit(1)


section("3. data_loader")
try:
    data = load_all(data_dir='.', n_ini=20, device=device, verbose=False)
    assert data['X_pde_pool'].device.type == device.split(':')[0]
    print(f"  {PASS} Data loaded on {device}")
    print(f"       PDE pool:    {tuple(data['X_pde_pool'].shape)}")
    print(f"       Seismograms: {tuple(data['X_seism'].shape)}")
    print(f"       Snapshot1:   {tuple(data['X_snap1'].shape)}")

    import numpy as np
    alpha = true_wavespeed_np(np.array([0.5,0.8]), np.array([0.1,0.3]))
    print(f"  {PASS} true_wavespeed  values={alpha.round(3)}")

except Exception as e:
    print(f"  {FAIL} {e}"); import traceback; traceback.print_exc(); sys.exit(1)


section("4. PINNModel — PDE residual")
try:
    cfg   = TrainConfig(hidden_layers=2, neurons=16, n_epochs=1,
                        device=device, out_dir='/tmp/pinn_torch_smoke')
    model = PINNModel(cfg, torch.device(device))
    X_pde = torch.rand(20, 3, device=device)
    X_pde[:, 0] *= float(data['X_pde_pool'][:, 0].max())
    X_pde[:, 1] *= float(data['X_pde_pool'][:, 1].max())
    X_pde[:, 2] *= 0.4
    R = model.pde_residual(X_pde)
    assert R.shape == (20, 1), R.shape
    print(f"  {PASS} PDE residual computed  shape={tuple(R.shape)}")

    X_s = data['X_snap1'][:10]
    ux, uz = model.disp_at(X_s)
    assert ux.shape == (10, 1)
    print(f"  {PASS} Displacement via autograd  ux={tuple(ux.shape)}")

    losses = compute_losses(model, X_pde, data)
    for k, v in losses.items():
        assert not torch.isnan(v), f"{k} is NaN"
    print(f"  {PASS} All losses computed: "
          + "  ".join(f"{k}={v.item():.3e}" for k, v in losses.items()))

except Exception as e:
    print(f"  {FAIL} {e}"); import traceback; traceback.print_exc(); sys.exit(1)


section("5. PINNTrainer — 5 epochs")
try:
    cfg = TrainConfig(
        name='smoke', n_epochs=5, batch_pde=200,
        hidden_layers=2, neurons=16,
        print_every=1, save_every=10,
        use_adaptive_weights=False, use_adaptive_colloc=False,
        use_fourier=False, device=device,
        out_dir='/tmp/pinn_torch_smoke', lr_decay=False,
    )
    trainer = PINNTrainer(cfg, data)
    history = trainer.train()
    ev = trainer.evaluate()
    assert not np.isnan(ev['alpha_rmse'])
    print(f"  {PASS} Trainer OK  α-RMSE={ev['alpha_rmse']:.4f}")

except Exception as e:
    print(f"  {FAIL} {e}"); import traceback; traceback.print_exc(); sys.exit(1)


section("6. All improvements — 5 epochs")
try:
    cfg2 = TrainConfig(
        name='smoke_full', n_epochs=5, batch_pde=200,
        hidden_layers=2, neurons=16, print_every=1, save_every=10,
        use_adaptive_weights=True, weight_update_every=2,
        use_adaptive_colloc=True,  resample_every=2,
        use_fourier=True, fourier_m=16, fourier_sigma=5.0,
        device=device, out_dir='/tmp/pinn_torch_smoke2', lr_decay=False,
    )
    t2 = PINNTrainer(cfg2, data)
    t2.train()
    print(f"  {PASS} All improvements combined — OK")

except Exception as e:
    print(f"  {FAIL} {e}"); import traceback; traceback.print_exc(); sys.exit(1)


section("All tests passed ✓")
dev_str = f"GPU ({torch.cuda.get_device_name(0)})" if device == 'cuda' else "CPU"
print(f"""
Device: {dev_str}

Run commands:
  # Quick demo (500 epochs, a few minutes on GPU):
  python run_experiment.py --device cuda --quick --out_dir results_quick/

  # Full training (30k epochs, ~1-3 hours on GPU):
  python run_experiment.py --device cuda --epochs 30000 --out_dir results/

  # Post-run analysis:
  python analyze_results.py --out_dir results/
""")
