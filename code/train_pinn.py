import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pinn_core import MLP, FourierEncoding
from data_loader import (true_wavespeed_np, LX, LZ, AX, AZ,
                          T_TOTAL, T_START,
                          INV_Z_ST, INV_Z_FI, INV_X_ST, INV_X_FI)


class TrainConfig:
    def __init__(
        self,
        hidden_layers        = 4,
        neurons              = 50,
        activation           = 'tanh',
        use_fourier          = False,
        fourier_m            = 64,
        fourier_sigma        = 5.0,
        n_epochs             = 30_000,
        batch_pde            = 10_000,
        lr                   = 1e-3,
        lr_decay             = True,
        print_every          = 1_000,
        save_every           = 5_000,
        lam_pde              = 0.1,
        lam_snap             = 1.0,
        lam_bc               = 0.1,
        lam_obs              = 1.0,
        use_adaptive_weights = True,
        weight_update_every  = 1_000,
        weight_alpha         = 0.1,
        use_adaptive_colloc  = True,
        resample_every       = 1_000,
        resample_frac        = 0.5,
        device               = 'cuda',
        seed                 = 42,
        out_dir              = 'results/baseline',
        name                 = 'baseline',
    ):
        self.__dict__.update(locals())
        del self.__dict__['self']

    @property
    def torch_device(self):
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU.")
            return torch.device('cpu')
        return torch.device(self.device)


class PINNModel(nn.Module):

    def __init__(self, cfg: TrainConfig, device):
        super().__init__()
        self.cfg    = cfg
        self.device = device

        ub_phi = torch.tensor([[AX/LX, AZ/LZ, T_TOTAL - T_START]],
                              dtype=torch.float32, device=device)
        ub_alp = torch.tensor([[AX/LX, AZ/LZ]],
                              dtype=torch.float32, device=device)

        fourier = None
        n_in    = 3
        if cfg.use_fourier:
            fourier = FourierEncoding(3, cfg.fourier_m, cfg.fourier_sigma).to(device)
            n_in    = fourier.d_out

        layers_phi = [n_in] + [cfg.neurons] * cfg.hidden_layers + [1]
        self.nn_phi = MLP(layers_phi, activation=cfg.activation,
                          fourier=fourier, ub=ub_phi).to(device)

        layers_alp = [2, 20, 20, 20, 20, 20, 1]
        self.nn_alp = MLP(layers_alp, activation='tanh',
                          fourier=None, ub=ub_alp).to(device)

    def phi(self, xzt):
        return self.nn_phi(xzt)

    def alpha(self, xz):
        alpha_star = torch.tanh(self.nn_alp(xz))
        x_km = xz[:, 0:1] * LX
        z_km = xz[:, 1:2] * LZ
        lld  = 1000.0
        mask = (torch.sigmoid(lld*(z_km - INV_Z_ST)) *
                torch.sigmoid(lld*(INV_Z_FI - z_km)) *
                torch.sigmoid(lld*(x_km - INV_X_ST)) *
                torch.sigmoid(lld*(INV_X_FI - x_km)))
        return 3.0 + 2.0 * alpha_star * mask

    def pde_residual(self, X):
        X = X.requires_grad_(True)
        phi  = self.phi(X)
        ones = torch.ones_like(phi)

        dphi = torch.autograd.grad(phi, X, grad_outputs=ones,
                                   create_graph=True)[0]

        phi_xx = torch.autograd.grad(dphi[:, 0:1], X, grad_outputs=ones,
                                     create_graph=True)[0][:, 0:1]
        phi_zz = torch.autograd.grad(dphi[:, 1:2], X, grad_outputs=ones,
                                     create_graph=True)[0][:, 1:2]
        phi_tt = torch.autograd.grad(dphi[:, 2:3], X, grad_outputs=ones,
                                     create_graph=True)[0][:, 2:3]

        alpha  = self.alpha(X[:, :2])
        R = phi_tt - alpha**2 * ((1/LX)**2 * phi_xx + (1/LZ)**2 * phi_zz)
        return R

    def disp_at(self, X):
        X = X.requires_grad_(True)
        phi  = self.phi(X)
        ones = torch.ones_like(phi)
        dphi = torch.autograd.grad(phi, X, grad_outputs=ones,
                                   create_graph=True)[0]
        ux = dphi[:, 0:1] / LX
        uz = dphi[:, 1:2] / LZ
        return ux, uz

    def pressure_laplacian(self, X):
        X = X.requires_grad_(True)
        phi  = self.phi(X)
        ones = torch.ones_like(phi)
        dphi = torch.autograd.grad(phi, X, grad_outputs=ones,
                                   create_graph=True)[0]
        phi_xx = torch.autograd.grad(dphi[:, 0:1], X, grad_outputs=ones,
                                     create_graph=True)[0][:, 0:1]
        phi_zz = torch.autograd.grad(dphi[:, 1:2], X, grad_outputs=ones,
                                     create_graph=True)[0][:, 1:2]
        return (1/LX)**2 * phi_xx + (1/LZ)**2 * phi_zz


def compute_losses(model, X_pde, data):
    d = data

    R_pde   = model.pde_residual(X_pde)
    loss_pde = torch.mean(R_pde**2)

    ux1, uz1 = model.disp_at(d['X_snap1'])
    ux2, uz2 = model.disp_at(d['X_snap2'])
    loss_snap = (torch.mean((ux1 - d['U_snap1_x'])**2) +
                 torch.mean((uz1 - d['U_snap1_z'])**2) +
                 torch.mean((ux2 - d['U_snap2_x'])**2) +
                 torch.mean((uz2 - d['U_snap2_z'])**2))

    lap_bc   = model.pressure_laplacian(d['X_bc'])
    loss_bc  = torch.mean(lap_bc**2)

    ux_s, uz_s = model.disp_at(d['X_seism'])
    loss_obs = (torch.mean((ux_s - d['Sx'])**2) +
                torch.mean((uz_s - d['Sz'])**2))

    return dict(pde=loss_pde, snap=loss_snap, bc=loss_bc, obs=loss_obs)


def update_lambdas(model, optimizer, X_pde, data, lam, alpha_ema=0.1):
    names   = ['pde', 'snap', 'bc', 'obs']
    g_norms = {}
    params  = list(model.parameters())

    for name in names:
        optimizer.zero_grad()
        losses = compute_losses(model, X_pde, data)
        losses[name].backward()
        with torch.no_grad():
            total_norm = sum(
                p.grad.abs().mean().item()
                for p in params if p.grad is not None
            )
        g_norms[name] = total_norm + 1e-12

    g_max    = max(g_norms.values())
    new_lam  = torch.tensor([g_max / g_norms[n] for n in names],
                             dtype=torch.float32, device=lam.device)

    lam_new  = alpha_ema * new_lam + (1 - alpha_ema) * lam
    optimizer.zero_grad()
    return lam_new


def resample_pde_points(model, pool, batch_size, resample_frac=0.5, probe=8000):
    idx_probe = torch.randperm(len(pool))[:probe]
    X_probe   = pool[idx_probe]

    R = model.pde_residual(X_probe.clone()).detach().abs().squeeze()

    probs   = R / (R.sum() + 1e-30)
    n_high  = int(batch_size * resample_frac)
    n_unif  = batch_size - n_high

    high_idx = torch.multinomial(probs, n_high, replacement=True)
    unif_idx = torch.randperm(len(pool), device=pool.device)[:n_unif]

    return torch.cat([X_probe[high_idx], pool[unif_idx]], dim=0)


class PINNTrainer:

    def __init__(self, cfg: TrainConfig, data: dict):
        self.cfg    = cfg
        self.data   = data
        self.device = cfg.torch_device

        torch.manual_seed(cfg.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(cfg.seed)

        self.data = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        self.model = PINNModel(cfg, self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        if cfg.lr_decay:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01)
        else:
            self.scheduler = None

        self.lam = torch.tensor([cfg.lam_pde, cfg.lam_snap,
                                  cfg.lam_bc, cfg.lam_obs],
                                 dtype=torch.float32, device=self.device)

        pool = self.data['X_pde_pool']
        idx  = torch.randperm(len(pool))[:cfg.batch_pde]
        self.X_pde = pool[idx].clone()

        os.makedirs(cfg.out_dir, exist_ok=True)
        self.history = {k: [] for k in
                        ['epoch','loss','pde','snap','bc','obs',
                         'lam_pde','lam_snap','lam_bc','lam_obs',
                         'alpha_rmse','lr']}

        n_phi = sum(p.numel() for p in self.model.nn_phi.parameters())
        n_alp = sum(p.numel() for p in self.model.nn_alp.parameters())
        print(f"[PINN] {cfg.name}  |  act={cfg.activation}  "
              f"fourier={cfg.use_fourier}  "
              f"adaptive_w={cfg.use_adaptive_weights}  "
              f"adaptive_c={cfg.use_adaptive_colloc}  "
              f"device={self.device}")
        print(f"[PINN] φ-params: {n_phi:,}    α-params: {n_alp:,}")

    def _step(self):
        self.optimizer.zero_grad()
        losses = compute_losses(self.model, self.X_pde, self.data)
        lam = self.lam
        L = (lam[0]*losses['pde'] + lam[1]*losses['snap'] +
             lam[2]*losses['bc']  + lam[3]*losses['obs'])
        L.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return L.item(), {k: v.item() for k, v in losses.items()}

    def evaluate(self):
        d = self.data
        xxs, zzs = d['xxs'], d['zzs']
        alpha_true = true_wavespeed_np(xxs * LX, zzs * LZ)

        N = xxs.size
        xz_eval = torch.tensor(
            np.c_[xxs.ravel(), zzs.ravel()].astype(np.float32),
            device=self.device)
        alpha_pred = self.model.alpha(xz_eval).detach().cpu().numpy().reshape(xxs.shape)

        alpha_rmse = float(np.sqrt(np.mean((alpha_pred - alpha_true)**2)))

        ux_t, uz_t = self.model.disp_at(d['X_test'])
        U_pred = torch.sqrt(ux_t**2 + uz_t**2).detach().cpu().numpy().reshape(xxs.shape)
        U_true = np.sqrt(d['U_test_x'].cpu().numpy()**2 +
                         d['U_test_z'].cpu().numpy()**2).reshape(xxs.shape)
        wf_err = float(np.mean(np.abs(U_pred - U_true)) /
                       (np.mean(np.abs(U_true)) + 1e-12))

        return dict(alpha_pred=alpha_pred, alpha_true=alpha_true,
                    U_pred=U_pred, U_true=U_true,
                    alpha_rmse=alpha_rmse, wavefield_rel_err=wf_err)

    def train(self):
        cfg = self.cfg
        t0  = time.time()

        print(f"\n{'─'*60}")
        print(f"  Training: {cfg.name}  ({cfg.n_epochs:,} epochs)")
        print(f"{'─'*60}")

        for epoch in range(cfg.n_epochs + 1):

            L, losses = self._step()
            if self.scheduler:
                self.scheduler.step()

            if (cfg.use_adaptive_weights and epoch > 0
                    and epoch % cfg.weight_update_every == 0):
                self.lam = update_lambdas(
                    self.model, self.optimizer, self.X_pde, self.data,
                    self.lam, cfg.weight_alpha)

            if (cfg.use_adaptive_colloc and epoch > 0
                    and epoch % cfg.resample_every == 0):
                self.X_pde = resample_pde_points(
                    self.model, self.data['X_pde_pool'],
                    cfg.batch_pde, cfg.resample_frac)

            elif not cfg.use_adaptive_colloc and epoch % 100 == 0:
                pool = self.data['X_pde_pool']
                idx  = torch.randperm(len(pool))[:cfg.batch_pde]
                self.X_pde = pool[idx].clone()

            if epoch % cfg.print_every == 0:
                ev      = self.evaluate()
                elapsed = time.time() - t0
                lr_now  = self.optimizer.param_groups[0]['lr']
                lam     = self.lam.cpu().numpy()

                print(f"  ep={epoch:6d}  L={L:.4e}  "
                      f"pde={losses['pde']:.3e}  snap={losses['snap']:.3e}  "
                      f"bc={losses['bc']:.3e}  obs={losses['obs']:.3e}  "
                      f"α-RMSE={ev['alpha_rmse']:.4f}  "
                      f"lr={lr_now:.2e}  t={elapsed:.0f}s")

                for k, v in zip(
                    ['epoch','loss','pde','snap','bc','obs',
                     'lam_pde','lam_snap','lam_bc','lam_obs',
                     'alpha_rmse','lr'],
                    [epoch, L, losses['pde'], losses['snap'],
                     losses['bc'], losses['obs'],
                     lam[0], lam[1], lam[2], lam[3],
                     ev['alpha_rmse'], lr_now]
                ):
                    self.history[k].append(float(v))

            if epoch % cfg.save_every == 0 and epoch > 0:
                self._save_plots(epoch)
                self._save_checkpoint(epoch)

        self._save_plots(cfg.n_epochs, final=True)
        self._save_history()
        print(f"\n[PINN] Done: {cfg.name}  →  {cfg.out_dir}")
        return self.history

    def _save_plots(self, epoch, final=False):
        cfg = self.cfg
        ev  = self.evaluate()
        d   = self.data
        xxs, zzs = d['xxs'], d['zzs']

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f'{cfg.name}  —  epoch {epoch:,}', fontsize=11)

        def _cf(ax, arr, title, cmap='jet', vmin=None, vmax=None):
            cf = ax.contourf(xxs*LX, zzs*LZ, arr, 50, cmap=cmap,
                             vmin=vmin, vmax=vmax)
            plt.colorbar(cf, ax=ax, shrink=0.75)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('x (km)', fontsize=7)
            ax.set_ylabel('z (km)', fontsize=7)
            ax.set_aspect('equal')

        vmin_a = ev['alpha_true'].min()
        vmax_a = ev['alpha_true'].max()

        _cf(axes[0,0], ev['alpha_true'], 'True α (km/s)',
            vmin=vmin_a, vmax=vmax_a)
        _cf(axes[0,1], ev['alpha_pred'], 'Predicted α (km/s)',
            vmin=vmin_a, vmax=vmax_a)
        _cf(axes[0,2], np.abs(ev['alpha_true'] - ev['alpha_pred']),
            f'|Δα|  RMSE={ev["alpha_rmse"]:.4f}', cmap='hot')
        _cf(axes[1,0], ev['U_true'],  'SpecFem |U|', cmap='seismic')
        _cf(axes[1,1], ev['U_pred'],  'PINN |U|',    cmap='seismic')
        _cf(axes[1,2], np.abs(ev['U_true'] - ev['U_pred']),
            f'|ΔU|  relErr={ev["wavefield_rel_err"]:.4f}', cmap='hot')

        fname = ('final' if final else f'ep{epoch:06d}') + '.png'
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.out_dir, fname), dpi=130)
        plt.close(fig)

    def _save_history(self):
        cfg = self.cfg
        h   = self.history
        np.save(os.path.join(cfg.out_dir, 'history.npy'), h)

        if not h['epoch']:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(cfg.name)

        ax = axes[0]
        for k, col in [('pde','r'),('snap','b'),('bc','k'),('obs','c'),('loss','--y')]:
            ax.semilogy(h['epoch'], h[k], col, label=k, lw=1.2)
        ax.set_xlabel('epoch'); ax.set_ylabel('loss')
        ax.legend(fontsize=7); ax.set_title('Loss components')

        ax = axes[1]
        ax.plot(h['epoch'], h['alpha_rmse'], 'b', lw=1.5)
        ax.set_xlabel('epoch'); ax.set_ylabel('α RMSE (km/s)')
        ax.set_title('Wavespeed RMSE')

        ax = axes[2]
        for k, col in [('lam_pde','r'),('lam_snap','b'),
                       ('lam_bc','k'),('lam_obs','c')]:
            ax.plot(h['epoch'], h[k], col, label=k, lw=1.2)
        ax.set_xlabel('epoch'); ax.set_ylabel('λ')
        ax.set_title('Loss weights'); ax.legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(cfg.out_dir, 'loss_history.png'), dpi=130)
        plt.close(fig)

    def _save_checkpoint(self, epoch):
        path = os.path.join(self.cfg.out_dir, f'checkpoint_ep{epoch:06d}.pt')
        torch.save({
            'epoch'      : epoch,
            'model_state': self.model.state_dict(),
            'opt_state'  : self.optimizer.state_dict(),
            'lam'        : self.lam,
            'history'    : self.history,
        }, path)
