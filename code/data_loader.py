import os
import numpy as np
import scipy.interpolate as interpolate
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NX, NZ        = 100, 100
N_ABS         = 10
AX_SPEC       = 1.5
AZ_SPEC       = 0.5
X_SEISM_SPEC  = 1.3

DX = AX_SPEC / NX
DZ = AZ_SPEC / NZ

LX = 3.0
LZ = 3.0

AX = X_SEISM_SPEC - N_ABS * DX
AZ = AZ_SPEC      - N_ABS * DZ

T_TOTAL  = 0.5
T_START  = 0.1
T_SERIES = 0.5

S_SPEC  = 5e-5
T01     = 2000 * S_SPEC
T02     = 2300 * S_SPEC
T_TEST  = 5000 * S_SPEC

N_SEISMO  = 20
Z0_SEISM  = AZ
ZL_SEISM  = 0.06 - N_ABS * DZ

U_SCALE   = 1.0 / 3640.0
SUBSAMPLE = 100

INV_Z_ST = 0.1  - N_ABS * DZ
INV_Z_FI = 0.45 - N_ABS * DZ
INV_X_ST = 0.7  - N_ABS * DX
INV_X_FI = 1.25 - N_ABS * DX


def _to_tensor(arr, device, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype, device=device)


def load_all(data_dir='.', n_ini=40, device='cpu', verbose=True):
    if verbose:
        print(f"[data] Loading from: {os.path.abspath(data_dir)}  →  device={device}")

    xx_plot, zz_plot = np.meshgrid(
        np.linspace(0, AX / LX, n_ini),
        np.linspace(0, AZ / LZ, n_ini)
    )

    grid_file = os.path.join(data_dir, 'event1', 'wavefields',
                             'wavefield_grid_for_dumps_000.txt')
    X0 = np.loadtxt(grid_file) / 1000.0
    X0[:, 0] /= LX
    X0[:, 1] /= LZ
    xz = X0[:, :2]

    wf_dir = os.path.join(data_dir, 'event1', 'wavefields')
    wfs = sorted(f for f in os.listdir(wf_dir) if f.startswith('wavefield0'))
    U0 = [np.loadtxt(os.path.join(wf_dir, f)) for f in wfs]

    xf = N_ABS * DX / LX
    zf = N_ABS * DZ / LZ
    xxs, zzs = np.meshgrid(
        np.linspace(xf, X_SEISM_SPEC / LX, n_ini),
        np.linspace(zf, AZ_SPEC / LZ, n_ini)
    )
    interp_pts = np.column_stack([xxs.ravel(), zzs.ravel()])

    def interp(U_raw):
        r = interpolate.griddata(xz, U_raw, interp_pts, fill_value=0.0)
        return r[:, 0:1] / U_SCALE, r[:, 1:2] / U_SCALE

    U1x, U1z = interp(U0[0])
    U2x, U2z = interp(U0[1])
    Utx, Utz = interp(U0[2])

    N = xxs.size
    ones_n = np.ones((N, 1))

    X_snap1 = np.c_[xxs.ravel(), zzs.ravel(), 0.0 * ones_n]
    X_snap2 = np.c_[xxs.ravel(), zzs.ravel(), (T02 - T01) * ones_n]
    X_test  = np.c_[xxs.ravel(), zzs.ravel(), (T_TEST - T01) * ones_n]

    seis_dir = os.path.join(data_dir, 'event1', 'seismograms')
    all_files = sorted(os.listdir(seis_dir))

    def load_comp(comp):
        files = sorted(f for f in all_files if f.endswith(f'BX{comp}.semd'))
        records = [np.loadtxt(os.path.join(seis_dir, f)) for f in files]
        t_raw = -records[0][0, 0] + records[0][:, 0]
        valid = np.where((t_raw >= T_START) & (t_raw <= T_SERIES))[0]
        idx   = valid[::SUBSAMPLE]
        t_sub = (t_raw[idx] - t_raw[idx[0]]).reshape(-1, 1)
        sig   = np.concatenate([r[idx, 1:2] / U_SCALE for r in records], axis=0)
        return sig, t_sub, len(records)

    Sz, t_sub, n_stat = load_comp('Z')
    Sx, _,     _      = load_comp('X')

    n_t = len(t_sub)
    d_s = abs(ZL_SEISM - Z0_SEISM) / (N_SEISMO - 1)
    Xseis_parts = []
    for i in range(n_stat):
        z_i = (Z0_SEISM - i * d_s) / LZ
        Xseis_parts.append(np.c_[
            np.full((n_t, 1), AX / LX),
            np.full((n_t, 1), z_i),
            t_sub
        ])
    X_seism = np.concatenate(Xseis_parts, axis=0)

    rng = np.random.default_rng(0)
    bcx = rng.random(100) * AX / LX
    bct = rng.random(50)  * (T_TOTAL - T_START)
    gx, gt = np.meshgrid(bcx, bct)
    X_bc = np.c_[gx.ravel(), np.full(gx.size, AZ / LZ), gt.ravel()]

    rng2 = np.random.default_rng(1)
    M = 200_000
    Xp = rng2.random((M, 3))
    Xp[:, 0] *= AX / LX
    Xp[:, 1] *= AZ / LZ
    Xp[:, 2] *= (T_TOTAL - T_START)

    def T(a):
        return _to_tensor(a.astype(np.float32), device)

    out = dict(
        X_pde_pool  = T(Xp),
        X_snap1     = T(X_snap1),   X_snap2    = T(X_snap2),
        U_snap1_x   = T(U1x),       U_snap1_z  = T(U1z),
        U_snap2_x   = T(U2x),       U_snap2_z  = T(U2z),
        X_seism     = T(X_seism),
        Sx          = T(Sx),        Sz         = T(Sz),
        X_bc        = T(X_bc),
        X_test      = T(X_test),
        U_test_x    = T(Utx),       U_test_z   = T(Utz),
        xxs         = xxs,          zzs        = zzs,
        domain      = dict(ax=AX, az=AZ, t_total=T_TOTAL, t_start=T_START,
                           Lx=LX, Lz=LZ),
    )

    if verbose:
        print(f"  PDE pool:    {Xp.shape[0]:,} pts")
        print(f"  Seismograms: {X_seism.shape[0]:,} pts  "
              f"({n_stat} stations × {n_t} time samples)")
        print(f"  Snapshots:   {n_ini}×{n_ini} = {N} pts each")

    return out


def true_wavespeed_np(x_km, z_km):
    cx = 1.0 - N_ABS * DX
    cz = 0.3 - N_ABS * DZ
    g  = (x_km - cx)**2 / 0.18**2 + (z_km - cz)**2 / 0.1**2
    return 3.0 - 0.25 * (1.0 + np.tanh(100.0 * (1.0 - g)))


def true_wavespeed_torch(x, z, device='cpu'):
    cx = torch.tensor(1.0 - N_ABS * DX, device=device)
    cz = torch.tensor(0.3 - N_ABS * DZ, device=device)
    g  = (x - cx)**2 / 0.18**2 + (z - cz)**2 / 0.1**2
    return 3.0 - 0.25 * (1.0 + torch.tanh(100.0 * (1.0 - g)))


def plot_inputs(data, save_dir='results/data_diagnostics'):
    os.makedirs(save_dir, exist_ok=True)
    xxs, zzs = data['xxs'], data['zzs']

    def _mag(vx, vz):
        return np.sqrt(vx.cpu().numpy()**2 + vz.cpu().numpy()**2).reshape(xxs.shape)

    panels = [
        (_mag(data['U_snap1_x'], data['U_snap1_z']), f'Snapshot 1  t={T01:.3f}s'),
        (_mag(data['U_snap2_x'], data['U_snap2_z']), f'Snapshot 2  t={T02:.3f}s'),
        (_mag(data['U_test_x'],  data['U_test_z']),  f'SpecFem test  t={T_TEST:.3f}s'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (mag, title) in zip(axes, panels):
        cf = ax.contourf(xxs * LX, zzs * LZ, mag, 60, cmap='jet')
        plt.colorbar(cf, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x (km)'); ax.set_ylabel('z (km)')
        ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'input_wavefields.png'), dpi=150)
    plt.close(fig)

    alpha_true = true_wavespeed_np(xxs * LX, zzs * LZ)
    fig, ax = plt.subplots(figsize=(5, 4))
    cf = ax.contourf(xxs * LX, zzs * LZ, alpha_true, 60, cmap='jet')
    plt.colorbar(cf, ax=ax)
    ax.set_title('True wavespeed α (km/s)'); ax.set_xlabel('x (km)'); ax.set_ylabel('z (km)')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'true_wavespeed.png'), dpi=150)
    plt.close(fig)
    print(f"[data] Diagnostic plots → {save_dir}/")
