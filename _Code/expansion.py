# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""expansion.py

A small, deterministic initial-condition generator that models cosmic expansion as an
iterative rescaling + finite-precision quantisation.

Design intent (toy model):
- Expansion is represented as N successive "doubling" steps on a fixed grid.
- Each step applies a gentle resampling (to couple to the lattice geometry) then quantises.
- Unbiased quantisation behaves like noise (~sqrt(N) * eps).
- Biased quantisation (e.g. floor/trunc) behaves like drift (~N * eps).

This module generates a small, correlated scalar perturbation field that can be mapped
into phi/vel for SG "primordial soup" style initial conditions.

No external deps (numpy only). Deterministic by default.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


QuantMode = Literal["nearest", "floor", "trunc"]
ResampleMode = Literal["none", "halfshift"]


def as_resample_mode(v: str) -> ResampleMode:
    if v == "none":
        return "none"
    if v == "halfshift":
        return "halfshift"
    raise ValueError(f"invalid resample mode: {v}")


def as_quant_mode(v: str) -> QuantMode:
    if v == "nearest":
        return "nearest"
    if v == "floor":
        return "floor"
    if v == "trunc":
        return "trunc"
    raise ValueError(f"invalid quant mode: {v}")


@dataclass(frozen=True)
class ExpansionParams:
    n: int = 192
    steps: int = 194
    bits: int = 24
    seed: int = 6174
    resample: ResampleMode = "halfshift"
    quant: QuantMode = "floor"
    mask_corr: float = 24.0
    mask_bias: float = 1.0
    mask_rms: float = 0.25
    core_exclude: int = 0
    sponge_exclude: int = 0
    out_dtype: str = "float32"


def _fail(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _quant_step(bits: int) -> float:
    _fail(bits >= 2, f"bits must be >=2, got {bits}")
    return float(2.0 ** (-int(bits)))


def _biased_quantize(x: np.ndarray, delta: float, mode: QuantMode) -> np.ndarray:
    if mode == "nearest":
        return np.round(x / delta) * delta
    if mode == "floor":
        return np.floor(x / delta) * delta
    if mode == "trunc":
        return np.trunc(x / delta) * delta
    raise ValueError(f"unknown quant mode: {mode}")


def _halfshift_resample(x: np.ndarray) -> np.ndarray:
    """A cheap, deterministic resampler that couples to lattice parity.

    We approximate a half-cell shift by averaging with a +1 roll along each axis.
    This is intentionally simple: the point is to create a consistent geometric coupling,
    not to be a high-quality interpolation kernel.
    """

    y = x
    y = 0.5 * (y + np.roll(y, shift=1, axis=0))
    y = 0.5 * (y + np.roll(y, shift=1, axis=1))
    y = 0.5 * (y + np.roll(y, shift=1, axis=2))
    return y


def _gaussian_kspace_filter(n: int, corr: float) -> np.ndarray:
    """Radial Gaussian low-pass in k-space.

    corr is a correlation length in voxels (roughly the 1-sigma scale in real space).
    """

    _fail(corr > 0.0, f"corr must be >0, got {corr}")
    kx = np.fft.fftfreq(n)
    ky = np.fft.fftfreq(n)
    kz = np.fft.fftfreq(n)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX * KX + KY * KY + KZ * KZ
    sigma = 1.0 / max(1e-9, float(corr))
    return np.exp(-0.5 * k2 / (sigma * sigma)).astype(np.float32)


def make_bias_mask(n: int, corr: float, rms: float, bias: float, seed: int) -> np.ndarray:
    """Create a smooth mask b(x) with non-zero mean (bias) and controlled RMS.

    Returns b(x) where mean(b) ~= bias and std(b) ~= rms.
    """

    _fail(n >= 8, f"n too small: {n}")
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal((n, n, n), dtype=np.float32)
    F = np.fft.fftn(noise)
    filt = _gaussian_kspace_filter(n, corr)
    smooth = np.fft.ifftn(F * filt).real.astype(np.float32)
    smooth -= float(np.mean(smooth))
    s = float(np.std(smooth))
    _fail(s > 0.0 and math.isfinite(s), "mask smoothing produced degenerate field")
    smooth /= s
    smooth *= float(rms)
    smooth += float(bias)
    return smooth


def expansion_drift_field(ep: ExpansionParams) -> Tuple[np.ndarray, dict]:
    """Generate a perturbation field T(x) via iterative resample + quantise.

    The returned field has zero mean (monopole removed) over the included region.
    """

    n = int(ep.n)
    _fail(n >= 8, f"n must be >=8, got {n}")
    steps = int(ep.steps)
    _fail(steps >= 1, f"steps must be >=1, got {steps}")
    delta = _quant_step(int(ep.bits))

    b = make_bias_mask(n=n, corr=float(ep.mask_corr), rms=float(ep.mask_rms), bias=float(ep.mask_bias), seed=int(ep.seed))

    T = np.zeros((n, n, n), dtype=np.float32)

    for _ in range(steps):
        if ep.resample == "halfshift":
            T = _halfshift_resample(T)
        elif ep.resample == "none":
            pass
        else:
            raise ValueError(f"unknown resample mode: {ep.resample}")
        T = _biased_quantize(T + delta * b, delta, ep.quant).astype(np.float32)

    cut = max(int(ep.core_exclude), int(ep.sponge_exclude), 0)
    if cut > 0:
        _fail(2 * cut < n, f"exclude too large for n={n}: cut={cut}")
        core = T[cut:-cut, cut:-cut, cut:-cut]
        core = core - float(np.mean(core))
        T2 = np.zeros_like(T)
        T2[cut:-cut, cut:-cut, cut:-cut] = core
        T = T2
    else:
        T = T - float(np.mean(T))

    info = {
        "n": n,
        "steps": steps,
        "bits": int(ep.bits),
        "delta": float(delta),
        "seed": int(ep.seed),
        "resample": str(ep.resample),
        "quant": str(ep.quant),
        "mask_corr": float(ep.mask_corr),
        "mask_bias": float(ep.mask_bias),
        "mask_rms": float(ep.mask_rms),
        "core_exclude": int(ep.core_exclude),
        "sponge_exclude": int(ep.sponge_exclude),
        "T_mean": float(np.mean(T)),
        "T_std": float(np.std(T)),
        "T_abs_max": float(np.max(np.abs(T))),
    }
    return T.astype(np.float32), info


def map_to_phi_vel(T: np.ndarray, phi_amp: float, vel_amp: float = 0.0, phi_bg: float = 0.0, seed: int = 6174) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Map a perturbation field into (phi, vel) initial conditions.

    - phi = phi_bg + phi_amp * T
    - vel = vel_amp * U where U is a correlated field derived from T (deterministic)

    vel is optional; default is 0.
    """

    _fail(T.ndim == 3 and T.shape[0] == T.shape[1] == T.shape[2], "T must be cubic 3D")
    phi = (float(phi_bg) + float(phi_amp) * T).astype(np.float32)
    if float(vel_amp) == 0.0:
        vel = np.zeros_like(phi)
        info = {"phi_amp": float(phi_amp), "vel_amp": 0.0, "phi_bg": float(phi_bg)}
        return phi, vel, info

    rng = np.random.default_rng(int(seed) ^ 0xA5A5)
    eta = rng.standard_normal(T.shape, dtype=np.float32)
    U = 0.7 * T + 0.3 * eta
    U -= float(np.mean(U))
    s = float(np.std(U))
    _fail(s > 0.0 and math.isfinite(s), "vel field degenerate")
    U /= s
    vel = (float(vel_amp) * U).astype(np.float32)
    info = {"phi_amp": float(phi_amp), "vel_amp": float(vel_amp), "phi_bg": float(phi_bg)}
    return phi, vel, info


# Public API for core.py: generate SG "bigbang" initial conditions.
def make_bigbang_ic(
    *,
    n: int,
    seed: int = 6174,
    levels: int = 194,
    bits: int = 24,
    quant: QuantMode = "floor",
    resample: ResampleMode = "halfshift",
    mask_corr: float = 24.0,
    mask_bias: float = 1.0,
    mask_rms: float = 0.25,
    core_exclude: int = 0,
    sponge_exclude: int = 0,
    phi_amp: float = 1.0,
    vel_amp: float = 0.0,
    phi_bg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate (phi, vel, info) initial conditions for SG runs.

    This is a convenience wrapper used by core.py when `--sg-bigbang` is enabled.

    The field is produced by iterative resample+quantise for `levels` steps with a biased
    mask. The resulting perturbation T is mapped to:
      phi = phi_bg + phi_amp * T
      vel = optional correlated field scaled by vel_amp (default 0).

    Returns:
      (phi, vel, info) where info includes both expansion and mapping provenance.
    """

    ep = ExpansionParams(
        n=int(n),
        steps=int(levels),
        bits=int(bits),
        seed=int(seed),
        resample=resample,
        quant=quant,
        mask_corr=float(mask_corr),
        mask_bias=float(mask_bias),
        mask_rms=float(mask_rms),
        core_exclude=int(core_exclude),
        sponge_exclude=int(sponge_exclude),
    )
    T, exp_info = expansion_drift_field(ep)
    phi, vel, map_info = map_to_phi_vel(
        T,
        phi_amp=float(phi_amp),
        vel_amp=float(vel_amp),
        phi_bg=float(phi_bg),
        seed=int(seed),
    )
    info = {
        "kind": "sg_bigbang",
        "expansion": exp_info,
        "mapping": map_info,
    }
    return phi, vel, info


def _rms_core(x: np.ndarray, cut: int) -> float:
    if cut <= 0:
        return float(np.sqrt(np.mean(x * x)))
    core = x[cut:-cut, cut:-cut, cut:-cut]
    return float(np.sqrt(np.mean(core * core)))


def diagnostic_scaling(n: int, bits: int, seed: int, quant: QuantMode, resample: ResampleMode, corr: float, rms: float, bias: float, max_pow2: int = 7, cut: int = 0) -> dict:
    """Check ~sqrt(N) vs ~N scaling by running steps = 2^k for k=3..max_pow2."""

    ks = list(range(3, int(max_pow2) + 1))
    Ns = [2 ** k for k in ks]
    vals = []
    for steps in Ns:
        ep = ExpansionParams(
            n=int(n),
            steps=int(steps),
            bits=int(bits),
            seed=int(seed),
            resample=resample,
            quant=quant,
            mask_corr=float(corr),
            mask_bias=float(bias),
            mask_rms=float(rms),
            core_exclude=int(cut),
            sponge_exclude=0,
        )
        T, _ = expansion_drift_field(ep)
        vals.append(_rms_core(T, cut))

    x = np.log(np.array(Ns, dtype=np.float64))
    y = np.log(np.array(vals, dtype=np.float64) + 1e-30)
    m = float(np.polyfit(x, y, 1)[0])
    return {"steps": Ns, "rms": vals, "slope_loglog": m}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expansion drift initial-condition generator (toy cosmology).")
    p.add_argument("--n", type=int, default=192)
    p.add_argument("--steps", type=int, default=194)
    p.add_argument("--bits", type=int, default=24)
    p.add_argument("--seed", type=int, default=6174)
    p.add_argument("--resample", type=str, default="halfshift", choices=["none", "halfshift"])
    p.add_argument("--quant", type=str, default="floor", choices=["nearest", "floor", "trunc"])
    p.add_argument("--mask-corr", type=float, default=24.0)
    p.add_argument("--mask-bias", type=float, default=1.0)
    p.add_argument("--mask-rms", type=float, default=0.25)
    p.add_argument("--core-exclude", type=int, default=0)
    p.add_argument("--phi-amp", type=float, default=1.0)
    p.add_argument("--vel-amp", type=float, default=0.0)
    p.add_argument("--phi-bg", type=float, default=0.0)
    p.add_argument("--save-npz", type=str, default="")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--diagnostic-max-pow2", type=int, default=7)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    a = _parse_args(argv)
    if bool(a.diagnostic):
        out = diagnostic_scaling(
            n=int(a.n),
            bits=int(a.bits),
            seed=int(a.seed),
            quant=as_quant_mode(str(a.quant)),
            resample=as_resample_mode(str(a.resample)),
            corr=float(a.mask_corr),
            rms=float(a.mask_rms),
            bias=float(a.mask_bias),
            max_pow2=int(a.diagnostic_max_pow2),
            cut=int(a.core_exclude),
        )
        print(f"[expansion] diagnostic slope_loglog={out['slope_loglog']:.3f} (0.5~noise, 1.0~drift)")
        print("[expansion] steps:", out["steps"])
        print("[expansion] rms:", [float(f"{v:.6g}") for v in out["rms"]])
        return

    ep = ExpansionParams(
        n=int(a.n),
        steps=int(a.steps),
        bits=int(a.bits),
        seed=int(a.seed),
        resample=as_resample_mode(str(a.resample)),
        quant=as_quant_mode(str(a.quant)),
        mask_corr=float(a.mask_corr),
        mask_bias=float(a.mask_bias),
        mask_rms=float(a.mask_rms),
        core_exclude=int(a.core_exclude),
        sponge_exclude=0,
    )

    T, info = expansion_drift_field(ep)
    phi, vel, map_info = map_to_phi_vel(T, phi_amp=float(a.phi_amp), vel_amp=float(a.vel_amp), phi_bg=float(a.phi_bg), seed=int(a.seed))

    print(f"[expansion] n={info['n']} steps={info['steps']} bits={info['bits']} delta={info['delta']:.3e} quant={info['quant']} resample={info['resample']}")
    print(f"[expansion] T: mean={info['T_mean']:.3e} std={info['T_std']:.3e} abs_max={info['T_abs_max']:.3e}")
    print(f"[expansion] phi: mean={float(np.mean(phi)):.3e} std={float(np.std(phi)):.3e} abs_max={float(np.max(np.abs(phi))):.3e}")
    print(f"[expansion] vel: mean={float(np.mean(vel)):.3e} std={float(np.std(vel)):.3e} abs_max={float(np.max(np.abs(vel))):.3e}")

    if str(a.save_npz):
        info_json = json.dumps(info, sort_keys=True)
        map_info_json = json.dumps(map_info, sort_keys=True)
        np.savez_compressed(
            str(a.save_npz),
            T=T.astype(np.float32),
            phi=phi.astype(np.float32),
            vel=vel.astype(np.float32),
            info_json=np.array(info_json, dtype=np.str_),
            map_info_json=np.array(map_info_json, dtype=np.str_),
        )
        print(f"[expansion] saved: {str(a.save_npz)}")


if __name__ == "__main__":
    main()