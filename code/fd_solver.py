import numpy as np
import os
import argparse
from pathlib import Path


def make_homogeneous_model(nx, nz, alpha_bg=3.0):
    return np.full((nz, nx), alpha_bg)


def make_gaussian_anomaly_model(nx, nz, dx, alpha_bg=3.0, alpha_min=2.0,
                                 cx_km=7.5, cz_km=7.5, sigma_km=2.5):
    x = np.arange(nx) * dx
    z = np.arange(nz) * dx
    X, Z = np.meshgrid(x, z)
    r2 = (X - cx_km) ** 2 + (Z - cz_km) ** 2
    alpha = alpha_bg - (alpha_bg - alpha_min) * np.exp(-r2 / (2 * sigma_km ** 2))
    return alpha


def make_ellipsoidal_anomaly_model(nx, nz, dx, alpha_bg=3.0, alpha_anom=2.0,
                                    cx_km=0.5, cz_km=0.25, a_km=0.2, b_km=0.1):
    x = np.arange(nx) * dx
    z = np.arange(nz) * dx
    X, Z = np.meshgrid(x, z)
    ellipse = ((X - cx_km) / a_km) ** 2 + ((Z - cz_km) / b_km) ** 2
    alpha = np.where(ellipse <= 1.0, alpha_anom, alpha_bg)
    return alpha


def ricker_wavelet(t, f0, t0=None):
    if t0 is None:
        t0 = 1.2 / f0
    u = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * u ** 2) * np.exp(-u ** 2)


def compute_displacement_from_phi(phi, dx):
    nz, nx = phi.shape
    ux = np.zeros_like(phi)
    uz = np.zeros_like(phi)

    ux[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dx)
    uz[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dx)

    ux[:, 0] = (phi[:, 1] - phi[:, 0]) / dx
    ux[:, -1] = (phi[:, -1] - phi[:, -2]) / dx
    uz[0, :] = (phi[1, :] - phi[0, :]) / dx
    uz[-1, :] = (phi[-1, :] - phi[-2, :]) / dx

    return ux, uz


def build_damping_field(nx, nz, n_sponge, damping_max=0.05, free_surface=True):
    damp = np.zeros((nz, nx))

    for i in range(n_sponge):
        strength = damping_max * ((n_sponge - i) / n_sponge) ** 2

        damp[:, i] = np.maximum(damp[:, i], strength)
        damp[:, -(i + 1)] = np.maximum(damp[:, -(i + 1)], strength)
        damp[-(i + 1), :] = np.maximum(damp[-(i + 1), :], strength)

        if not free_surface:
            damp[i, :] = np.maximum(damp[i, :], strength)

    return damp


def solve_acoustic_2d(alpha, dx, dt, nt, source_pos, source_wavelet,
                      receiver_z_idx=0, n_receivers=20,
                      snapshot_times=None, n_sponge=20,
                      free_surface=True, verbose=True):
    nz, nx = alpha.shape

    if snapshot_times is None:
        snapshot_times = []

    alpha_sq = alpha ** 2
    damping = build_damping_field(nx, nz, n_sponge, free_surface=free_surface)

    rx_start = n_sponge + 5
    rx_end = nx - n_sponge - 5
    receiver_x_idx = np.linspace(rx_start, rx_end, n_receivers, dtype=int)

    phi_prev = np.zeros((nz, nx))
    phi_curr = np.zeros((nz, nx))
    phi_next = np.zeros((nz, nx))

    snapshots = {}
    seismograms = np.zeros((nt, n_receivers))
    ux_seismograms = np.zeros((nt, n_receivers))
    uz_seismograms = np.zeros((nt, n_receivers))

    src_iz, src_ix = source_pos

    for it in range(nt):
        laplacian = np.zeros_like(phi_curr)

        laplacian[1:-1, 1:-1] = (
            phi_curr[1:-1, 2:] + phi_curr[1:-1, :-2]
            + phi_curr[2:, 1:-1] + phi_curr[:-2, 1:-1]
            - 4.0 * phi_curr[1:-1, 1:-1]
        ) / dx ** 2

        laplacian[2:-2, 2:-2] = (
            (-phi_curr[2:-2, 4:] + 16 * phi_curr[2:-2, 3:-1]
             - 30 * phi_curr[2:-2, 2:-2]
             + 16 * phi_curr[2:-2, 1:-3] - phi_curr[2:-2, :-4])
            + (-phi_curr[4:, 2:-2] + 16 * phi_curr[3:-1, 2:-2]
               - 30 * phi_curr[2:-2, 2:-2]
               + 16 * phi_curr[1:-3, 2:-2] - phi_curr[:-4, 2:-2])
        ) / (12.0 * dx ** 2)

        phi_next[1:-1, 1:-1] = (
            2.0 * phi_curr[1:-1, 1:-1]
            - phi_prev[1:-1, 1:-1]
            + dt ** 2 * alpha_sq[1:-1, 1:-1] * laplacian[1:-1, 1:-1]
        )

        phi_next[src_iz, src_ix] += dt ** 2 * alpha_sq[src_iz, src_ix] * source_wavelet[it] / (dx ** 2)

        phi_next *= (1.0 - damping)

        if free_surface:
            phi_next[0, :] = phi_next[1, :]

        seismograms[it, :] = phi_next[receiver_z_idx, receiver_x_idx]

        rz = receiver_z_idx
        for ri, rx in enumerate(receiver_x_idx):
            if 0 < rx < nx - 1:
                ux_seismograms[it, ri] = (phi_next[rz, rx + 1] - phi_next[rz, rx - 1]) / (2.0 * dx)
            elif rx == 0:
                ux_seismograms[it, ri] = (phi_next[rz, 1] - phi_next[rz, 0]) / dx
            else:
                ux_seismograms[it, ri] = (phi_next[rz, -1] - phi_next[rz, -2]) / dx
            if 0 < rz < nz - 1:
                uz_seismograms[it, ri] = (phi_next[rz + 1, rx] - phi_next[rz - 1, rx]) / (2.0 * dx)
            elif rz == 0:
                if nz > 2:
                    uz_seismograms[it, ri] = (-3*phi_next[0, rx] + 4*phi_next[1, rx] - phi_next[2, rx]) / (2.0 * dx)
                else:
                    uz_seismograms[it, ri] = (phi_next[1, rx] - phi_next[0, rx]) / dx
            else:
                uz_seismograms[it, ri] = (phi_next[-1, rx] - phi_next[-2, rx]) / dx

        if it in snapshot_times:
            snapshots[it] = phi_curr.copy()
            if verbose:
                print(f"  Saved snapshot at timestep {it} (t = {it * dt:.4f} s)")

        phi_prev, phi_curr = phi_curr, phi_next
        phi_next = np.zeros_like(phi_curr)

        if verbose and it % (nt // 10) == 0:
            print(f"  Step {it}/{nt}  max|φ| = {np.max(np.abs(phi_curr)):.6e}")

    return snapshots, seismograms, receiver_x_idx, ux_seismograms, uz_seismograms


class SimulationConfig:
    def __init__(self, name, velocity_model_fn, velocity_model_kwargs,
                 domain_x_km, domain_z_km, dx_km, t_total_s, f0_hz,
                 source_x_km, source_z_km,
                 snapshot_t1_s, snapshot_t2_s,
                 n_receivers=20, n_sponge=20, free_surface=True):
        self.name = name
        self.velocity_model_fn = velocity_model_fn
        self.velocity_model_kwargs = velocity_model_kwargs
        self.domain_x_km = domain_x_km
        self.domain_z_km = domain_z_km
        self.dx_km = dx_km
        self.t_total_s = t_total_s
        self.f0_hz = f0_hz
        self.source_x_km = source_x_km
        self.source_z_km = source_z_km
        self.snapshot_t1_s = snapshot_t1_s
        self.snapshot_t2_s = snapshot_t2_s
        self.n_receivers = n_receivers
        self.n_sponge = n_sponge
        self.free_surface = free_surface


CONFIGS = {
    "gaussian_forward": SimulationConfig(
        name="gaussian_forward",
        velocity_model_fn=make_gaussian_anomaly_model,
        velocity_model_kwargs=dict(
            alpha_bg=3.0, alpha_min=2.0,
            cx_km=7.5, cz_km=7.5, sigma_km=2.5,
        ),
        domain_x_km=15.0,
        domain_z_km=15.0,
        dx_km=0.05,
        t_total_s=5.0,
        f0_hz=2.0,
        source_x_km=7.5,
        source_z_km=0.1,
        snapshot_t1_s=0.0,
        snapshot_t2_s=0.1,
        n_receivers=20,
        n_sponge=20,
        free_surface=False,
    ),

    "homogeneous_inverse": SimulationConfig(
        name="homogeneous_inverse",
        velocity_model_fn=make_homogeneous_model,
        velocity_model_kwargs=dict(alpha_bg=3.0),
        domain_x_km=1.0,
        domain_z_km=0.5,
        dx_km=0.005,
        t_total_s=0.4,
        f0_hz=20.0,
        source_x_km=0.1,
        source_z_km=0.25,
        snapshot_t1_s=0.0,
        snapshot_t2_s=0.01,
        n_receivers=20,
        n_sponge=15,
        free_surface=True,
    ),

    "ellipsoidal_inverse": SimulationConfig(
        name="ellipsoidal_inverse",
        velocity_model_fn=make_ellipsoidal_anomaly_model,
        velocity_model_kwargs=dict(
            alpha_bg=3.0, alpha_anom=2.0,
            cx_km=0.5, cz_km=0.25, a_km=0.2, b_km=0.1,
        ),
        domain_x_km=1.0,
        domain_z_km=0.5,
        dx_km=0.005,
        t_total_s=0.4,
        f0_hz=20.0,
        source_x_km=0.1,
        source_z_km=0.25,
        snapshot_t1_s=0.0,
        snapshot_t2_s=0.01,
        n_receivers=20,
        n_sponge=15,
        free_surface=True,
    ),
}


def run_simulation(config: SimulationConfig, output_dir: str = "data"):
    print(f"\n{'=' * 60}")
    print(f"  Running: {config.name}")
    print(f"{'=' * 60}")

    nx = int(config.domain_x_km / config.dx_km) + 1
    nz = int(config.domain_z_km / config.dx_km) + 1
    dx = config.dx_km

    print(f"  Grid: {nx} x {nz} (dx = {dx} km)")

    alpha = config.velocity_model_fn(nx, nz, dx=dx, **config.velocity_model_kwargs) \
        if 'dx' in config.velocity_model_fn.__code__.co_varnames \
        else config.velocity_model_fn(nx, nz, **config.velocity_model_kwargs)

    alpha_max = np.max(alpha)
    print(f"  Velocity range: [{np.min(alpha):.2f}, {alpha_max:.2f}] km/s")

    safety = 0.5
    cfl_dt = dx / (alpha_max * np.sqrt(2.0))
    dt = safety * cfl_dt
    nt = int(config.t_total_s / dt) + 1
    t = np.arange(nt) * dt

    print(f"  dt = {dt:.6e} s  (CFL limit: {cfl_dt:.6e} s)")
    print(f"  nt = {nt}  (t_total = {config.t_total_s} s)")

    src_ix = int(config.source_x_km / dx)
    src_iz = int(config.source_z_km / dx)
    src_ix = np.clip(src_ix, 0, nx - 1)
    src_iz = np.clip(src_iz, 0, nz - 1)
    wavelet = ricker_wavelet(t, config.f0_hz)

    print(f"  Source at grid ({src_iz}, {src_ix}), f0 = {config.f0_hz} Hz")

    snap_it1 = int(config.snapshot_t1_s / dt)
    snap_it2 = int(config.snapshot_t2_s / dt)
    snapshot_times = [snap_it1, snap_it2]

    print(f"  Snapshots at t = {snap_it1 * dt:.4f} s, {snap_it2 * dt:.4f} s")

    snapshots, seismograms, receiver_x_idx, ux_seis, uz_seis = solve_acoustic_2d(
        alpha=alpha, dx=dx, dt=dt, nt=nt,
        source_pos=(src_iz, src_ix),
        source_wavelet=wavelet,
        receiver_z_idx=1 if config.free_surface else 0,
        n_receivers=config.n_receivers,
        snapshot_times=snapshot_times,
        n_sponge=config.n_sponge,
        free_surface=config.free_surface,
        verbose=True,
    )

    save_dir = os.path.join(output_dir, config.name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    x_coords = np.arange(nx) * dx
    z_coords = np.arange(nz) * dx
    t_coords = t
    receiver_x_km = receiver_x_idx * dx

    np.savez(os.path.join(save_dir, "velocity_model.npz"),
             alpha=alpha, x=x_coords, z=z_coords)

    snap1_phi = snapshots.get(snap_it1, np.zeros_like(alpha))
    snap2_phi = snapshots.get(snap_it2, np.zeros_like(alpha))
    snap1_ux, snap1_uz = compute_displacement_from_phi(snap1_phi, dx)
    snap2_ux, snap2_uz = compute_displacement_from_phi(snap2_phi, dx)

    np.savez(os.path.join(save_dir, "snapshots.npz"),
             snapshot_1=snap1_phi, snapshot_2=snap2_phi,
             snapshot_1_ux=snap1_ux, snapshot_1_uz=snap1_uz,
             snapshot_2_ux=snap2_ux, snapshot_2_uz=snap2_uz,
             t1=snap_it1 * dt, t2=snap_it2 * dt,
             x=x_coords, z=z_coords)

    np.savez(os.path.join(save_dir, "seismograms.npz"),
             seismograms=seismograms,
             ux_seismograms=ux_seis, uz_seismograms=uz_seis,
             t=t_coords, receiver_x=receiver_x_km,
             receiver_z=z_coords[1 if config.free_surface else 0])

    np.savez(os.path.join(save_dir, "metadata.npz"),
             nx=nx, nz=nz, dx=dx, dt=dt, nt=nt,
             t_total=config.t_total_s, f0=config.f0_hz,
             source_x=config.source_x_km, source_z=config.source_z_km,
             alpha_max=alpha_max, alpha_min=np.min(alpha))

    print(f"\n  Saved to {save_dir}/")
    print(f"  Snapshot 1 max|φ| = {np.max(np.abs(snapshots.get(snap_it1, [0]))):.6e}")
    print(f"  Snapshot 2 max|φ| = {np.max(np.abs(snapshots.get(snap_it2, [0]))):.6e}")
    print(f"  Seismogram max|φ| = {np.max(np.abs(seismograms)):.6e}")

    return snapshots, seismograms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ground truth data for PINN experiments."
    )
    parser.add_argument(
        "case", nargs="?", default="all",
        choices=list(CONFIGS.keys()) + ["all"],
        help="Which simulation case to run (default: all).",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Base output directory (default: data/).",
    )
    args = parser.parse_args()

    if args.case == "all":
        for name, cfg in CONFIGS.items():
            run_simulation(cfg, output_dir=args.output_dir)
    else:
        run_simulation(CONFIGS[args.case], output_dir=args.output_dir)

    print("\nDone.")
