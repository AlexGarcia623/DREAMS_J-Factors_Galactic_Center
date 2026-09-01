"""J-factor toy model

In this module I use a simple toy model to study how the
J-factor sky map seen from the solar circle depends on:

    1. The shape of the DM halo (spherical / triaxial / cored).
    2. The misalignment (tilt) between the halo's principal axes and the
       Galactic disk.
    3. A physical displacement of the cusp away from the potential minimum.
    4. The observer's azimuth on the solar circle.

The line-of-sight integrals are JIT-compiled and vectorised with JAX. The
observer sits in the disk plane (z = 0); the local sky frame has e3 pointing
toward the Galactic Centre, e1 along the rotation-pole direction (zhat) and
e2 = e3 x e1, so theta_x maps to the e1 component of the LOS and theta_y to
the e2 component.

Convenzioni
-----------
    a >= b >= c, with major axis along x and minor along z.
    q = b / a (intermediate-to-major), s = c / a (minor-to-major).
    Tilt rotates the principal axes in the xz plane: positive tilt rotates
    the minor axis from zhat toward xhat (i.e. partly along the LOS to the GC).
    Centroid: J-weighted mean of (theta_x, theta_y) over the sky map.

How you run this stuff
---
    python jfactor_toy.py             # produce all figures
    python jfactor_toy.py --only maps qs   # produce a subset
    python jfactor_toy.py --list      # list available figures
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

import cmasher as cmr

plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 2.25*1.25
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = 'true'
plt.rcParams['ytick.minor.visible'] = 'true'
plt.rcParams['xtick.major.width'] = 1.5*1.25
plt.rcParams['ytick.major.width'] = 1.5*1.25
plt.rcParams['xtick.minor.width'] = 1.0*1.25
plt.rcParams['ytick.minor.width'] = 1.0*1.25
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.minor.size'] = 4.5
plt.rcParams['xtick.top']   = True
plt.rcParams['ytick.right'] = True


# ---------------------------------------------------------------------------
# Halo and density profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Halo:
    """Halo main parameters. My defaults are typical MW-like (NFW, c=15, M200 = 1e12)."""
    M_200: float = 1e12          # M_sun
    c_200: float = 15.0
    rho_crit: float = 127.4      # M_sun / kpc^3 at z = 0
    r_core: float = 0.05         # numerical softening on r [kpc]
    R_obs: float = 8.0           # solar radius [kpc]

    @property
    def r_200(self) -> float:
        return (3.0 * self.M_200 / (4.0 * np.pi * 200.0 * self.rho_crit)) ** (1.0 / 3.0)

    @property
    def r_s(self) -> float:
        return self.r_200 / self.c_200

    @property
    def rho_s(self) -> float:
        c = self.c_200
        return self.M_200 / (4.0 * np.pi * self.r_s ** 3
                             * (np.log(1.0 + c) - c / (1.0 + c)))

    def rho_at_R_obs(self) -> float:
        u = self.R_obs / self.r_s
        return self.rho_s / (u * (1.0 + u) ** 2)


DensityFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


def nfw(halo: Halo, q: float = 1.0, s: float = 1.0,
        tilt_deg: float = 0.0,
        dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> DensityFn:
    """Triaxial NFW with optional principal-axis tilt and centre offset.

    The tilt rotates the principal frame by ``tilt_deg`` about y, so the
    minor (z) axis acquires a component along x (partly along the LOS to
    the GC). With (dx, dy, dz) != 0 the cusp sits at that offset from the
    origin; this is the toy analogue of a dynamical displacement similar to what we and Kuhlen observe.
    """
    rs, rhos, rcore = halo.r_s, halo.rho_s, halo.r_core
    ct = jnp.cos(jnp.deg2rad(tilt_deg))
    st = jnp.sin(jnp.deg2rad(tilt_deg))

    @jit
    def rho(x, y, z):
        xx, yy, zz = x - dx, y - dy, z - dz
        xr = xx * ct + zz * st
        zr = -xx * st + zz * ct
        r_ell = jnp.sqrt(xr ** 2 + (yy / q) ** 2 + (zr / s) ** 2 + rcore ** 2)
        u = r_ell / rs
        return rhos / (u * (1.0 + u) ** 2)
    return rho


def burkert(halo: Halo, r_b: float, q: float = 1.0, s: float = 1.0,
            tilt_deg: float = 0.0) -> DensityFn:
    """Cored Burkert profile, normalised to match NFW(R_obs)."""
    rho_local = halo.rho_at_R_obs()
    x_obs = halo.R_obs / r_b
    rho_0 = rho_local * (1.0 + x_obs) * (1.0 + x_obs ** 2)
    ct = jnp.cos(jnp.deg2rad(tilt_deg))
    st = jnp.sin(jnp.deg2rad(tilt_deg))

    @jit
    def rho(x, y, z):
        xr = x * ct + z * st
        zr = -x * st + z * ct
        r_ell = jnp.sqrt(xr ** 2 + (y / q) ** 2 + (zr / s) ** 2)
        u = r_ell / r_b
        return rho_0 / ((1.0 + u) * (1.0 + u ** 2))
    return rho


def nfw_rdep(halo: Halo,
             q_in: float, s_in: float, q_out: float, s_out: float,
             r_trans: float, tilt_deg: float = 0.0) -> DensityFn:
    """NFW with axis ratios that interpolate from (q_in, s_in) at the
    centre to (q_out, s_out) at large radii on the scale ``r_trans``.
    """
    rs, rhos, rcore = halo.r_s, halo.rho_s, halo.r_core
    ct = jnp.cos(jnp.deg2rad(tilt_deg))
    st = jnp.sin(jnp.deg2rad(tilt_deg))

    @jit
    def rho(x, y, z):
        # |r| is rotation-invariant, so the spherical radius can be evaluated
        # in the lab frame.
        r_sph = jnp.sqrt(x ** 2 + y ** 2 + z ** 2 + rcore ** 2)
        t = (r_sph / r_trans) ** 2
        q = q_out + (q_in - q_out) / (1.0 + t)
        s = s_out + (s_in - s_out) / (1.0 + t)
        xr = x * ct + z * st
        zr = -x * st + z * ct
        r_ell = jnp.sqrt(xr ** 2 + (y / q) ** 2 + (zr / s) ** 2 + rcore ** 2)
        u = r_ell / rs
        return rhos / (u * (1.0 + u) ** 2)
    return rho


# ---------------------------------------------------------------------------
# J-factor engine
# ---------------------------------------------------------------------------

def _local_frame(obs: jnp.ndarray):
    """Right-handed sky basis (e1, e2, e3) with e3 toward the origin (GC).

    ``e1`` is the projection of zhat orthogonal to e3, so for any observer
    in the disk plane it coincides with zhat. The 0.99 cutoff handles the
    edge case |obs . zhat| ~ 1, which never triggers for observers on the
    solar circle but keeps the function well-defined off-plane.
    """
    e3 = -obs / jnp.linalg.norm(obs)
    z_hat = jnp.array([0.0, 0.0, 1.0])
    ref = jnp.where(jnp.abs(jnp.dot(e3, z_hat)) > 0.99,
                    jnp.array([1.0, 0.0, 0.0]), z_hat)
    e1 = ref - jnp.dot(ref, e3) * e3
    e1 = e1 / jnp.linalg.norm(e1)
    e2 = jnp.cross(e3, e1)
    return e1, e2, e3

def pixel_solid_angle(TX: np.ndarray, TY: np.ndarray) -> np.ndarray:
    """True solid angle [sr] subtended by each pixel of a regular
    (theta_x, theta_y) grid under the zenithal-equidistant projection used
    by ``make_jmap`` (psi = sqrt(thx^2+thy^2) is the true angle from the
    map centre). Assumes TX, TY come from a regular meshgrid (as produced
    by ``_grid``).
    """
    dpix_deg = abs(TX[0, 1] - TX[0, 0])
    dpix_rad = np.deg2rad(dpix_deg)
    psi = np.deg2rad(np.hypot(TX, TY))
    jac = np.where(psi > 1e-9, np.sin(psi) / psi, 1.0)  # -> 1 at psi = 0
    return jac * dpix_rad ** 2


def to_dJdOmega(jmap: np.ndarray, TX: np.ndarray, TY: np.ndarray) -> np.ndarray:
    """Convert a per-pixel J map into the differential dJ/dOmega [same
    J-units / sr], dividing by the true pixel solid angle."""
    return jmap / pixel_solid_angle(TX, TY)

def make_jmap(rho_fn: DensityFn, lmax: float = 300.0, nlos: int = 1000):
    """Build a JIT'd J-factor sky-map evaluator for a given density.

    Returns a function ``compute(obs, theta_x, theta_y) -> J`` that takes
    arrays ``theta_x``, ``theta_y`` in degrees and returns the LOS integral
    of rho^2 in M_sun^2 / kpc^5. ``obs`` is the observer's Cartesian
    position in kpc.
    """
    l_vals = jnp.linspace(0.0, lmax, nlos)
    dl = l_vals[1] - l_vals[0]

    @jit
    def _single_los(obs, nhat):
        pts = obs[None, :] + l_vals[:, None] * nhat[None, :]
        rho = rho_fn(pts[:, 0], pts[:, 1], pts[:, 2])
        return jnp.trapezoid(rho ** 2, dx=dl)

    _batch = jit(vmap(vmap(_single_los, in_axes=(None, 0)), in_axes=(None, 0)))

    def compute(obs_np, thx, thy, differential: bool = False):
        obs = jnp.asarray(obs_np)
        e1, e2, e3 = _local_frame(obs)
        px = jnp.deg2rad(jnp.asarray(thx))
        py = jnp.deg2rad(jnp.asarray(thy))
        psi = jnp.sqrt(px ** 2 + py ** 2)
        phi = jnp.arctan2(py, px)
        nhat = (jnp.sin(psi)[..., None] * jnp.cos(phi)[..., None] * e1
                + jnp.sin(psi)[..., None] * jnp.sin(phi)[..., None] * e2
                + jnp.cos(psi)[..., None] * e3)
        out = np.asarray(_batch(obs, nhat))
        if differential:
            out = out / np.asarray(pixel_solid_angle(np.asarray(thx), np.asarray(thy)))
        return out
    return compute


def make_jann(rho_fn: DensityFn, npsi: int = 25, nphi: int = 72,
              lmax: float = 300.0, nlos: int = 800):
    """Build an evaluator returning the solid-angle-averaged J in an annulus."""
    jmap = make_jmap(rho_fn, lmax=lmax, nlos=nlos)
    PSI_grid, PHI_grid = np.meshgrid(np.linspace(0.0, 1.0, npsi),
                                       np.linspace(0.0, 2.0 * np.pi, nphi,
                                                   endpoint=False))

    def compute(obs_np, psi_min_deg, psi_max_deg):
        psi_min = np.deg2rad(psi_min_deg)
        psi_max = np.deg2rad(psi_max_deg)
        PSI = PSI_grid * (psi_max - psi_min) + psi_min
        thx = np.rad2deg(PSI) * np.cos(PHI_grid)
        thy = np.rad2deg(PSI) * np.sin(PHI_grid)
        return float(np.average(jmap(obs_np, thx, thy), weights=np.sin(PSI)))
    return compute


# ---------------------------------------------------------------------------
# Map post-processing -- a crude simulation analogue
# ---------------------------------------------------------------------------

def flatten_and_noise(jmap, TX, TY, flat_deg: float = 2.0,
                      noise_frac: float = 0.05,
                      rng: np.random.Generator | None = None):
    """Cap J inside ``flat_deg`` at the ring-mean and add multiplicative
    Gaussian noise inside the cap. Mimics finite resolution: simulations
    cannot reproduce the cusp, and what is left there is dominated by noise
    at the level of the resolution-limited density. 

    Note: when called repeatedly with the same ``rng`` each call advances
    the RNG state, so successive evaluations are independent realisations.
    """
    if rng is None:
        rng = np.random.default_rng()
    r = np.hypot(TX, TY)
    dpix = abs(TX[0, 1] - TX[0, 0])
    ring = np.abs(r - flat_deg) < dpix
    if ring.sum() > 0:
        j_ring = float(jmap[ring].mean())
    else:
        j_ring = float(np.median(jmap[r < flat_deg + dpix]))

    inner = r < flat_deg
    flat = jmap.copy()
    flat[inner] = np.minimum(jmap[inner], j_ring)

    noisy = flat.copy()
    sigma = noise_frac * flat[inner]
    noisy[inner] = np.maximum(flat[inner] + rng.normal(0.0, sigma), 0.0)
    return flat, noisy


def centroid(jmap, TX, TY) -> tuple[float, float]:
    """J-weighted centroid, in degrees. Not the peak location."""
    w = jmap / jmap.sum()
    return float(np.sum(w * TX)), float(np.sum(w * TY))
    
def peak_location(jmap, TX, TY) -> tuple[float, float]:
    """Location of the maximum peak, in degrees."""
    # Find the flat index of the maximum value, then convert to 2D indices
    row, col = np.unravel_index(np.argmax(jmap), jmap.shape)
    
    # Return the coordinates at that specific peak location
    return float(TX[row, col]), float(TY[row, col])

def top_percentile_location(jmap, TX, TY, top_pct: float = 1.0) -> tuple[float, float]:
    """Unweighted mean location of the top ``top_pct`` percent brightest
    pixels, in degrees. A compromise between ``centroid`` (weighted mean
    over the whole map, biased toward the box center) and ``peak_location``
    (single noisiest pixel).
    """
    thresh = np.percentile(jmap, 100.0 - top_pct)
    mask = jmap >= thresh
    return float(np.mean(TX[mask])), float(np.mean(TY[mask]))

def offset(jmap, TX, TY) -> float:
    # cx, cy = centroid(jmap, TX, TY)
    cx, cy = peak_location(jmap, TX, TY)
    # cx, cy = top_percentile_location(jmap, TX, TY)
    return float(np.hypot(cx, cy))


# ---------------------------------------------------------------------------
# Plotting helpers 
# ---------------------------------------------------------------------------

# A minimal, dependency-free style override. For the publication style used
# elsewhere in the project, ``import plot_params; rcParams.update(...)``.
_RC = {
    "axes.labelsize": 16,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "savefig.bbox": "tight",
    "savefig.dpi": 180,
}


def _try_project_style() -> None:
    """Use the project's plot_params if available; otherwise fall back to _RC (I have my own plot_params locally which I like)."""
    try:
        import plot_params  # noqa: F401
        plt.rcParams.update(plot_params.params)
    except Exception:
        plt.rcParams.update(_RC)


def _psi_circles(ax, radii=(2, 5, 8), color="w", **kw):
    for rd in radii:
        ax.add_patch(Circle((0, 0), rd, fill=False, color=color,
                             ls="--", lw=1.0, alpha=0.6, **kw))


def _solar_circle_obs(R: float, phi: float) -> np.ndarray:
    return np.array([R * np.cos(phi), R * np.sin(phi), 0.0])


def _grid(npix: int, span_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    th1d = np.linspace(-span_deg, span_deg, npix)
    TX, TY = np.meshgrid(th1d, th1d)
    return th1d, TX, TY


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_density_profiles(halo: Halo, q: float, s: float,
                          r_b_values: Sequence[float],
                          outdir: Path) -> None:
    """NFW (spherical and along principal axes) plus Burkert profiles for
    several core radii. Burkert curves are normalised to match NFW at R_odot.
    """
    r = np.logspace(-1.0, np.log10(halo.r_200), 500)
    jr = jnp.array(r)
    z0 = jnp.zeros_like(jr)

    rho_sph = nfw(halo)
    rho_tri = nfw(halo, q=q, s=s)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.loglog(r, np.asarray(rho_sph(jr, z0, z0)), lw=2.5, label="NFW spherical")
    ax.loglog(r, np.asarray(rho_tri(jr, z0, z0)), "--", lw=2,
              label=fr"NFW triaxial, along $x$ (major)")
    ax.loglog(r, np.asarray(rho_tri(z0, jr, z0)), "-.", lw=2,
              label=fr"NFW triaxial, along $y$ ($q={q}$)")
    ax.loglog(r, np.asarray(rho_tri(z0, z0, jr)), ":", lw=2.5,
              label=fr"NFW triaxial, along $z$ ($s={s}$)")
    for r_b in r_b_values:
        rho_b = burkert(halo, r_b)
        ax.loglog(r, np.asarray(rho_b(jr, z0, z0)), lw=2.0,
                  label=fr"Burkert ($r_b = {r_b}$ kpc)")

    ax.axvline(halo.R_obs, color="grey", ls="--", alpha=0.4, lw=1.2)
    ax.axvline(halo.r_s, color="grey", ls=":", alpha=0.4, lw=1.2)
    ax.text(halo.R_obs * 1.08, 2e7, r"$R_\odot$", fontsize=13, color="grey")
    ax.text(halo.r_s * 1.12, 2e7, r"$r_s$", fontsize=13, color="grey")
    ax.set(xlabel=r"$r$ [kpc]",
           ylabel=r"$\rho\,[M_\odot\,\mathrm{kpc}^{-3}]$",
           xlim=(0.1, halo.r_200),
           title="NFW vs Burkert density profiles")
    ax.legend(loc="lower left", fontsize=11)
    fig.savefig(outdir / "fig_density.png")
    plt.close(fig)


def fig_jmap_panel(halo: Halo, q: float, s: float, tilt_deg: float,
                    npix: int, span_deg: float, outdir: Path) -> None:
    """Three J-factor sky maps: spherical, triaxial-aligned, triaxial-tilted.
    Same colour range across panels for direct comparison.
    """
    th1d, TX, TY = _grid(npix, span_deg)
    obs = _solar_circle_obs(halo.R_obs, 0.0)

    maps = {}
    for name, rho_fn in [("Spherical", nfw(halo)),
                          (fr"Triaxial ($q={q},\,s={s}$)", nfw(halo, q, s)),
                          (fr"Triaxial + tilt ${tilt_deg:.0f}^\circ$",
                           nfw(halo, q, s, tilt_deg=tilt_deg))]:
        maps[name] = make_jmap(rho_fn)(obs, TX, TY)

    vmin = min(np.log10(j).min() for j in maps.values())
    vmax = max(np.log10(j).max() for j in maps.values())

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for ax, (title, jm) in zip(axes, maps.items()):
        cx, cy = centroid(jm, TX, TY)
        im = ax.pcolormesh(th1d, th1d, np.log10(jm), cmap="inferno",
                           shading="gouraud", vmin=vmin, vmax=vmax)
        _psi_circles(ax)
        ax.plot(0, 0, "+", color="white", ms=14, mew=2)
        ax.plot(cx, cy, "x", color="limegreen", ms=14, mew=2.5)
        ax.set(xlabel=r"$\theta_x$ [deg]", ylabel=r"$\theta_y$ [deg]",
               title=f"{title}\noffset = {np.hypot(cx, cy):.2f}$^\\circ$")
        ax.set_aspect("equal")

    fig.subplots_adjust(right=0.92)
    cax = fig.add_axes([0.935, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cax, label=r"$\log_{10}\,J$")
    fig.suptitle(f"J-factor sky maps -- observer at "
                 f"$(R_\\odot, 0, 0) = ({halo.R_obs}, 0, 0)$ kpc",
                 y=0.99, fontsize=15)
    fig.savefig(outdir / "fig_jmap_panel.png")
    plt.close(fig)


def alex_plot1(halo: Halo, q: float, s: float, tilt_deg: float,
                    npix: int, span_deg: float, outdir: Path) -> None:
    """Three J-factor sky maps: spherical, triaxial-aligned, triaxial-tilted.
    Same colour range across panels for direct comparison.
    """
    th1d, TX, TY = _grid(npix, span_deg)
    obs = _solar_circle_obs(halo.R_obs, 0.0)

    maps = {}
    for name, rho_fn in [("Spherical", nfw(halo)),
                         ("Spherical offset", nfw(halo, dz=float(0.296))),
                          ("Triaxial offset", nfw(halo, 0.833, 0.719, dz=float(0.296), tilt_deg=float(45)))]:
        maps[name] = make_jmap(rho_fn)(obs, TX, TY, differential=True)

    vmin = min(np.log10(j).min() for j in maps.values())
    vmax = max(np.log10(j).max() for j in maps.values())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    texts = [
        r'${\rm Spherical~NFW}$',
        r'${\rm Shifted~NFW}$',
        r'${\rm Shifted~NFW}+{\rm Triaxial}$'
    ]
    
    for iii, (ax, (title, jm)) in enumerate(zip(axes, maps.items())):
        # cx, cy = centroid(jm, TX, TY)
        cx, cy = peak_location(jm, TX, TY)
        # cx, cy = top_percentile_location(jm, TX, TY)
        im = ax.pcolormesh(th1d, th1d, np.log10(jm), cmap="cmr.dusk",
                           vmin=vmin, vmax=vmax, rasterized=True)
        # _psi_circles(ax)
        # ax.plot(0, 0, "+", color="white", ms=14, mew=2)
        ax.plot(cx, cy, "+", color="k", ms=14, mew=2.5)
        # ax.set(xlabel=r"$\theta_x$ [deg]", ylabel=r"$\theta_y$ [deg]",
        #        title=f"{title}\noffset = {np.hypot(cx, cy):.2f}$^\\circ$")
        ax.text(0.025,0.925,texts[iii], transform=ax.transAxes, color='w')
        ax.text(0.025,0.85,r'${\rm Offset}$' + fr'$~= {np.hypot(cx, cy):.2f}^\circ$', transform=ax.transAxes, color='w')
        if iii == 1:
            ax.text(0.975,0.0625,r'$d_{\rho_{\rm max}\to\Phi_{\rm min}}=0.296~{\rm kpc}$', transform=ax.transAxes,
                    color='w', ha='right')
        if iii == 2:
            ax.text(0.975,0.10,r'$q_{\rm 3D}=0.833$', transform=ax.transAxes,
                    color='w', ha='right')
            ax.text(0.975,0.025,r'$s_{\rm 3D}=0.719$', transform=ax.transAxes,
                    color='w', ha='right')
        ax.set_aspect("equal")

        ax.axhline(0.0, color='white', ls='--', lw=0.5, alpha=0.5)
        ax.axvline(0.0, color='white', ls='--', lw=0.5, alpha=0.5)
    
    fig.subplots_adjust(wspace=0.04)
    cax = fig.add_axes([0.93, 0.15, 0.016, 0.7])
    fig.colorbar(
        im, 
        cax=cax, 
        label=r'$\log({\rm d}{\cal J}/{\rm d}\Omega~{\rm [GeV^2\,cm^{-5}]})$', 
        extend='both',
        extendfrac=0.05
    )
    
    fig.savefig(outdir / "alex1.pdf", bbox_inches='tight')
    plt.close(fig)

def fig_j_vs_azimuth(halo: Halo, q: float, s: float,
                      tilt_deg_list: Sequence[float],
                      psi_range: tuple[float, float],
                      outdir: Path) -> None:
    """Mean J in an angular annulus as a function of observer azimuth on the
    solar circle, for spherical, triaxial, and several tilted halos.
    """
    n = 48
    phis = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    def scan(rho_fn):
        jann = make_jann(rho_fn)
        return np.array([jann(_solar_circle_obs(halo.R_obs, p), *psi_range)
                          for p in phis])

    series = {"Spherical": scan(nfw(halo)),
              fr"Triaxial ($q={q},\,s={s}$)": scan(nfw(halo, q, s))}
    for tilt in tilt_deg_list:
        series[fr"Triaxial + tilt ${tilt:.0f}^\circ$"] = scan(nfw(halo, q, s, tilt_deg=tilt))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, vals in series.items():
        ax.plot(np.rad2deg(phis), vals / vals.mean(),
                marker="o", ms=3, lw=1.8, label=label)
    ax.axhline(1.0, color="grey", ls=":", alpha=0.4)
    ax.set(xlabel=r"Observer azimuth $\phi_{\rm obs}$ [deg]",
           ylabel=r"$J / \langle J \rangle$",
           title=fr"J in {psi_range[0]}$^\circ$--{psi_range[1]}$^\circ$ annulus")
    ax.legend(ncol=2)
    fig.savefig(outdir / "fig_J_vs_azimuth.png")
    plt.close(fig)

    for label, vals in series.items():
        pk2pk = (vals.max() - vals.min()) / vals.mean() * 100.0
        print(f"  {label:38s}  peak-to-peak = {pk2pk:5.2f}%")


def fig_offset_vs_tilt(halo: Halo, q: float, s: float,
                        r_b_values: Sequence[float],
                        npix: int, span_deg: float,
                        outdir: Path) -> None:
    """Centroid offset vs principal-axis tilt for NFW and Burkert (with
    several core radii). The two zeros at tilt = 0 and 90 are the residual
    mirror symmetries of the projected density.
    """
    th1d, TX, TY = _grid(npix, span_deg)
    obs = _solar_circle_obs(halo.R_obs, 0.0)
    tilts = np.arange(0.0, 91.0, 3.0)

    def scan(profile_factory):
        out = []
        for tilt in tilts:
            rho_fn = profile_factory(tilt)
            jm = make_jmap(rho_fn)(obs, TX, TY)
            out.append(offset(jm, TX, TY))
        return np.array(out)

    off_nfw = scan(lambda t: nfw(halo, q, s, tilt_deg=float(t)))
    off_burk = {r_b: scan(lambda t, r_b=r_b:
                            burkert(halo, r_b, q, s, tilt_deg=float(t)))
                for r_b in r_b_values}

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(tilts, off_nfw, "o-", ms=5, lw=2.2, label="NFW (cuspy)")
    for r_b, off in off_burk.items():
        ax.plot(tilts, off, "s-", ms=5, lw=2.0,
                label=fr"Burkert ($r_b = {r_b}$ kpc)")
    ax.set(xlabel="Halo tilt [deg]",
           ylabel="Centroid offset from GC [deg]",
           title=fr"Centroid offset vs tilt ($q={q},\,s={s}$)")
    ax.legend()
    fig.savefig(outdir / "fig_offset_vs_tilt.png")
    plt.close(fig)

    print(f"  NFW  max offset = {off_nfw.max():.3f} deg at tilt = "
          f"{tilts[off_nfw.argmax()]:.0f} deg")
    for r_b, off in off_burk.items():
        print(f"  Burkert r_b={r_b:.1f} kpc  max = {off.max():.3f} deg at "
              f"tilt = {tilts[off.argmax()]:.0f} deg")


def fig_qs_scan(halo: Halo, tilt_deg: float, npix: int, span_deg: float,
                 nq: int, ns: int, with_flat_noise: bool,
                 outdir: Path) -> None:
    """Centroid offset over the (q, s) plane at fixed tilt. ``with_flat_noise``
    additionally produces the simulation-analogue version (flattened cusp +
    Gaussian noise inside 2 deg) and a side-by-side comparison. Gaussian noise (asked by Jonah) 
    doesn't seem super relevant (if at the % level).
    """
    th1d, TX, TY = _grid(npix, span_deg)
    obs = _solar_circle_obs(halo.R_obs, 0.0)
    q_arr = np.linspace(0.5, 1.0, nq)
    s_arr = np.linspace(0.3, 1.0, ns)

    raw = np.full((ns, nq), np.nan)
    flat = np.full((ns, nq), np.nan)
    rng = np.random.default_rng(42)

    total = sum(1 for q in q_arr for s in s_arr if s <= q)
    done = 0
    for iq, q in enumerate(q_arr):
        for js, s in enumerate(s_arr):
            if s > q:
                continue
            done += 1
            jm = make_jmap(nfw(halo, q, s, tilt_deg=tilt_deg))(obs, TX, TY)
            raw[js, iq] = offset(jm, TX, TY)
            if with_flat_noise:
                jm_f, _ = flatten_and_noise(jm, TX, TY, rng=rng)
                flat[js, iq] = offset(jm_f, TX, TY)
            print(f"  ({done}/{total})  q={q:.2f}  s={s:.2f}  "
                  f"raw={raw[js, iq]:.3f} deg", end="\r")
    print()

    panels = [("Raw NFW", raw)]
    if with_flat_noise:
        panels.append(("Flattened ($<2^\\circ$)", flat))

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 6.5),
                              squeeze=False)
    vmax = max(np.nanmax(g) for _, g in panels)
    for ax, (title, grid) in zip(axes[0], panels):
        masked = np.ma.masked_invalid(grid)
        im = ax.pcolormesh(q_arr, s_arr, masked, cmap="magma_r",
                           shading="gouraud", vmin=0, vmax=vmax)
        cs = ax.contour(q_arr, s_arr, masked,
                        levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        colors="white", linewidths=1.0, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=10, fmt="%.1f$^\\circ$")
        ax.plot([0.3, 1.0], [0.3, 1.0], "w--", lw=1.5, alpha=0.7,
                label=r"$s = q$")
        ax.set(xlabel=r"$q = b/a$",
               ylabel=r"$s = c/a$",
               xlim=(q_arr[0], q_arr[-1]),
               ylim=(s_arr[0], s_arr[-1]),
               title=title)
        ax.legend(loc="upper left")
    fig.colorbar(im, ax=axes[0], label="Centroid offset [deg]", shrink=0.85)
    fig.suptitle(fr"(q, s) scan at tilt $= {tilt_deg:.0f}^\circ$", y=1.02)
    fig.savefig(outdir / "fig_qs_scan.png")
    plt.close(fig)

    idx = np.unravel_index(np.nanargmax(raw), raw.shape)
    print(f"  raw max = {np.nanmax(raw):.3f} deg at q={q_arr[idx[1]]:.2f}, "
          f"s={s_arr[idx[0]]:.2f}")


def alex_plot2(halo: Halo, q: float, s: float, tilt_deg: float,
                         d_arr: np.ndarray, npix: int, span_deg: float,
                         outdir: Path) -> None:
    """Centroid offset vs a physical displacement of the cusp along z
    (perpendicular to the LOS). Compares the spherical case (pure projection)
    against the extreme triaxial case at the peak tilt.
    """
    th1d, TX, TY = _grid(npix, span_deg)
    obs = _solar_circle_obs(halo.R_obs, 0.0)
    rng = np.random.default_rng(42)

    off_sph  = np.empty_like(d_arr)
    off_tri1 = np.empty_like(d_arr)
    off_tri2 = np.empty_like(d_arr)
    off_tri3 = np.empty_like(d_arr)

    for i, d in enumerate(d_arr):
        jm_s  = make_jmap(nfw(halo, dz=float(d)))(obs, TX, TY, differential=True)
        jm_t1 = make_jmap(nfw(halo, 0.725, 0.578, tilt_deg=135, dz=float(d)))(obs, TX, TY, differential=True)
        jm_t2 = make_jmap(nfw(halo, 0.883, 0.719, tilt_deg=135, dz=float(d)))(obs, TX, TY, differential=True)
        jm_t3 = make_jmap(nfw(halo, 0.958, 0.840, tilt_deg=135, dz=float(d)))(obs, TX, TY, differential=True)

        off_sph[i]  = offset(jm_s, TX, TY)
        off_tri1[i] = offset(jm_t1, TX, TY)
        off_tri2[i] = offset(jm_t2, TX, TY)
        off_tri3[i] = offset(jm_t3, TX, TY)
        
        print(f"  d={d * 1e3:4.0f} pc  ", end="\r")
    print()

    fig, ax = plt.subplots(figsize=(6, 5))

    colors = cmr.take_cmap_colors('cmr.dusk', 10)
    
    ax.plot(d_arr, off_sph, "-", lw=2.0, color=colors[2],
            label=r"${\rm Spherical}$")
    ax.plot(d_arr, off_tri3, "--", lw=2.2, color=colors[4],
            label=r"${\rm Triaxial}~ (q=%0.3f,\,s=%0.3f)$" %(0.958, 0.840))
    ax.plot(d_arr, off_tri2, "-.", lw=2.2, color=colors[6],
            label=r"${\rm Triaxial}~ (q=%0.3f,\,s=%0.3f)$" %(0.883, 0.719))
    ax.plot(d_arr, off_tri1, ":", lw=2.2, color=colors[8],
            label=r"${\rm Triaxial}~ (q=%0.3f,\,s=%0.3f)$" %(0.725, 0.578))

    # ax.axvline(0.178, color='k', ls='-')
    # ax.axvline(0.450, color='k', ls='-')

    # ax.axhline(0.9, color='k', ls='-')
    # ax.axhline(0.6, color='k', ls='-')
    
    ax.set_xlabel(r'$d_{\rho_{\rm max}\to\Phi_{\rm min}}~[{\rm kpc}]$')
    ax.set_ylabel(r'${\rm Maximum~Offset~[degrees]}$')
    
    leg = ax.legend(loc="upper left", frameon=False)
    lCol = [colors[2], colors[4], colors[6], colors[8]]
    # lCol = [colors[2], colors[6]]
    for n, text in enumerate( leg.texts ):
        text.set_color( lCol[n] )   
        
    fig.savefig(outdir / "alex2.pdf")
    plt.close(fig)

def fig_physical_offset(halo: Halo, q: float, s: float, tilt_deg: float,
                         d_arr: np.ndarray, npix: int, span_deg: float,
                         outdir: Path) -> None:
    """Centroid offset vs a physical displacement of the cusp along z
    (perpendicular to the LOS). Compares the spherical case (pure projection)
    against the extreme triaxial case at the peak tilt.
    """
    th1d, TX, TY = _grid(npix, span_deg)
    obs = _solar_circle_obs(halo.R_obs, 0.0)
    rng = np.random.default_rng(42)

    off_sph = np.empty_like(d_arr)
    off_tri = np.empty_like(d_arr)
    off_tri_flat = np.empty_like(d_arr)

    for i, d in enumerate(d_arr):
        jm_s = make_jmap(nfw(halo, dz=float(d)))(obs, TX, TY)
        jm_t = make_jmap(nfw(halo, q, s, tilt_deg=tilt_deg, dz=float(d)))(obs, TX, TY)
        jm_t_f, _ = flatten_and_noise(jm_t, TX, TY, rng=rng)
        off_sph[i] = offset(jm_s, TX, TY)
        off_tri[i] = offset(jm_t, TX, TY)
        off_tri_flat[i] = offset(jm_t_f, TX, TY)
        print(f"  d={d * 1e3:4.0f} pc  sph={off_sph[i]:.2f}  "
              f"tri={off_tri[i]:.2f}  flat={off_tri_flat[i]:.2f} deg",
              end="\r")
    print()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(d_arr * 1e3, off_sph, "o--", ms=5, lw=2.0,
            label="Spherical (pure projection)")
    ax.plot(d_arr * 1e3, off_tri, "s-", ms=5, lw=2.2,
            label=fr"Triaxial ($q={q},\,s={s}$, tilt $={tilt_deg:.0f}^\circ$)")
    ax.plot(d_arr * 1e3, off_tri_flat, "^-", ms=5, lw=2.2,
            label="Triaxial + flattened ($<2^\\circ$)")
    ax.axvspan(300.0, 400.0, alpha=0.12, color="grey")
    ax.text(350.0, ax.get_ylim()[0] + 0.05, "Kuhlen+13",
            ha="center", fontsize=10, color="grey", style="italic")
    ax.set(xlabel="Physical cusp displacement $d_z$ [pc]",
           ylabel="J-factor centroid offset [deg]",
           title="Geometric + physical contributions")
    ax.legend(loc="upper left")
    fig.savefig(outdir / "fig_physical_offset.png")
    plt.close(fig)


def fig_observer_azimuth(halo: Halo, q: float, s: float, tilt_deg: float,
                          d_phys: float, npix: int, span_deg: float,
                          outdir: Path) -> None:
    """Centroid offset as a function of observer azimuth on the solar circle as asked me by Alex.
    Geometric only and geometric + physical displacement (along z), plus the
    side-on view of where on the orbit the offset is largest.
    """
    th1d, TX, TY = _grid(npix, span_deg)
    n = 48
    phis = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    rho_geo = nfw(halo, q, s, tilt_deg=tilt_deg)
    rho_phy = nfw(halo, q, s, tilt_deg=tilt_deg, dz=d_phys)
    jmap_geo = make_jmap(rho_geo)
    jmap_phy = make_jmap(rho_phy)

    off_geo = np.empty(n)
    off_phy = np.empty(n)
    cx_phy = np.empty(n)
    cy_phy = np.empty(n)
    for i, p in enumerate(phis):
        ob = _solar_circle_obs(halo.R_obs, p)
        off_geo[i] = offset(jmap_geo(ob, TX, TY), TX, TY)
        cxp, cyp = centroid(jmap_phy(ob, TX, TY), TX, TY)
        off_phy[i] = float(np.hypot(cxp, cyp))
        cx_phy[i] = cxp
        cy_phy[i] = cyp

    phi_deg = np.rad2deg(phis)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.plot(phi_deg, off_geo, "o-", ms=4, lw=2,
            label="Geometric only")
    ax.plot(phi_deg, off_phy, "s-", ms=4, lw=2,
            label=f"+ $d_z = {d_phys * 1e3:.0f}$ pc")
    ax.set(xlabel=r"Observer azimuth $\phi_{\rm obs}$ [deg]",
           ylabel="Centroid offset [deg]",
           title=fr"$q={q},\,s={s},\,$ tilt $= {tilt_deg:.0f}^\circ$")
    ax.legend()

    ax = axes[1]
    orbit = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot(halo.R_obs * np.cos(orbit), halo.R_obs * np.sin(orbit),
            color="grey", alpha=0.3, lw=1.2)
    ax.add_patch(Ellipse((0, 0), 2 * halo.r_s, 2 * halo.r_s * q,
                          fill=False, color="grey", ls=":", alpha=0.5))
    ax.plot(0, 0, "x", color="grey", ms=10, mew=2)
    sc = ax.scatter(halo.R_obs * np.cos(phis), halo.R_obs * np.sin(phis),
                    c=off_phy, cmap="magma_r", s=60, zorder=5,
                    edgecolors="k", linewidths=0.5)
    fig.colorbar(sc, ax=ax, label="Centroid offset [deg]", shrink=0.8)
    ax.set(xlabel="$x$ [kpc]", ylabel="$y$ [kpc]",
           xlim=(-12, 12), ylim=(-12, 12),
           title="Offset on the solar circle")
    ax.set_aspect("equal")

    fig.savefig(outdir / "fig_observer_azimuth.png")
    plt.close(fig)

    print(f"  geometric  min/max = {off_geo.min():.3f} / {off_geo.max():.3f} deg")
    print(f"  + d_z      min/max = {off_phy.min():.3f} / {off_phy.max():.3f} deg")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

FIGURES: dict[str, str] = {
    "density":  "NFW vs Burkert density profiles",
    "maps":     "Spherical / triaxial / tilted J-maps",
    "azimuth":  "J(annulus) vs observer azimuth",
    "tilt":     "Centroid offset vs tilt for NFW and Burkert",
    "qs":       "(q, s) parameter scan at fixed tilt",
    "physical": "Geometric + physical displacement",
    "observer": "Centroid offset vs observer azimuth",
}


def _resolve_outdir(path_arg: str) -> Path:
    out = Path(path_arg).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--outdir", default="./figs/andrea/", help="output directory")
    parser.add_argument("--only", nargs="+", choices=list(FIGURES),
                        help="only produce the listed figures")
    parser.add_argument("--list", action="store_true",
                        help="list available figures and exit")
    parser.add_argument("--npix", type=int, default=400,
                        help="sky-map resolution (per side)")
    parser.add_argument("--span", type=float, default=10.0,
                        help="sky-map half-extent in degrees")
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--s", type=float, default=0.3)
    parser.add_argument("--tilt", type=float, default=00.0,
                        help="default tilt for the J-map panel and (q, s) scan")
    parser.add_argument("--no-flat", action="store_true",
                        help="skip the flattened+noise variant in the (q, s) scan")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list:
        for k, v in FIGURES.items():
            print(f"  {k:10s}  {v}")
        return

    _try_project_style()
    outdir = _resolve_outdir(args.outdir)
    halo = Halo()
    selected = set(args.only) if args.only else set(FIGURES)

    print(f"r_200 = {halo.r_200:.1f} kpc, r_s = {halo.r_s:.2f} kpc, "
          f"rho_s = {halo.rho_s:.2e} M_sun/kpc^3")
    print(f"Outputs -> {outdir}")

    r_b_values = (3.0, 8.0, 15.0)
    t0 = time.time()

    selected = ['alex']
    
    if "density" in selected:
        print("\n[density]")
        fig_density_profiles(halo, args.q, args.s, r_b_values, outdir)

    if "maps" in selected:
        print("\n[maps]")
        fig_jmap_panel(halo, args.q, args.s, args.tilt,
                        args.npix, args.span, outdir)

    if "alex" in selected:
        alex_plot1(halo, args.q, args.s, args.tilt,
                        args.npix, args.span, outdir)

        alex_plot2(halo, q=args.q, s=args.s, tilt_deg=args.tilt,
                   d_arr=np.linspace(0.0, 1.0, 40),
                   npix=args.npix, span_deg=args.span,
                   outdir=outdir)
    
    if "azimuth" in selected:
        print("\n[azimuth]")
        fig_j_vs_azimuth(halo, args.q, args.s, (20.0, 35.0, 45.0),
                          (2.0, 8.0), outdir)

    if "tilt" in selected:
        print("\n[tilt]")
        fig_offset_vs_tilt(halo, args.q, args.s, r_b_values,
                            args.npix, args.span, outdir)

    if "qs" in selected:
        print("\n[qs]")
        fig_qs_scan(halo, args.tilt, args.npix, args.span,
                     nq=15, ns=15,
                     with_flat_noise=not args.no_flat, outdir=outdir)

    if "physical" in selected:
        print("\n[physical]")
        fig_physical_offset(halo, q=args.q, s=args.s, tilt_deg=args.tilt,
                             d_arr=np.linspace(0.0, 1.0, 40),
                             npix=args.npix, span_deg=args.span,
                             outdir=outdir)

    if "observer" in selected:
        print("\n[observer]")
        fig_observer_azimuth(halo, q=0.50, s=0.30, tilt_deg=21.0,
                              d_phys=0.35,
                              npix=args.npix, span_deg=args.span,
                              outdir=outdir)

    print(f"\nDone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
