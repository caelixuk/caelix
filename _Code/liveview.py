# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""liveview.py — CAELIX LiveView (consumer)

Runs as a separate process. Attaches to a shared memory block created by the simulation
and visualises the current 3D field as a volume.

Contract:
  - Never imports simulation modules.
  - Fail-soft on missing shared memory for a short grace period.
  - Close shared memory handle on exit.

Usage:
  python liveview.py --n 128
  python liveview.py --n 128 --shm-name CAELIX_shm
"""

from __future__ import annotations

import argparse
import os
import time
import traceback

import numpy as np

try:
    import pyvista as pv
    from pyvista import ImageData
    # Allow creating actors from empty PolyData; we update points on the first tick.
    try:
        pv.global_theme.allow_empty_mesh = True  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "liveview.py requires pyvista. Install it in your env before using --liveview."
    ) from e

from multiprocessing import shared_memory


def _lv_log(path: str | None, msg: str) -> None:
    if path is None:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip("\n") + "\n")
    except Exception:
        pass


def run_viewer(
    n: int,
    *,
    shm_name: str = "CAELIX_shm",
    connect_s: float = 5.0,
    fps: int = 30,
    log_path: str | None = None,
) -> None:
    shape = (int(n), int(n), int(n))
    if any(x <= 0 for x in shape):
        raise ValueError(f"liveview: invalid n={n}")

    _lv_log(
        log_path,
        f"[liveview] start pid={os.getpid()} n={n} shm={shm_name} fps={fps} connect_s={connect_s}",
    )

    deadline = time.time() + float(connect_s)
    shm = None
    while time.time() < deadline:
        try:
            try:
                shm = shared_memory.SharedMemory(
                    name=str(shm_name), track=False  # type: ignore[call-arg]
                )  # py3.12+
            except TypeError:
                shm = shared_memory.SharedMemory(name=str(shm_name))
            break
        except FileNotFoundError:
            time.sleep(0.1)

    if shm is None:
        _lv_log(
            log_path,
            f"[liveview] attach failed shm={shm_name} within {connect_s:.1f}s",
        )
        print(
            f"[liveview] could not attach to shared memory '{shm_name}' within {connect_s:.1f}s"
        )
        return

    try:
        phi_shm = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

        # Display view (may be decimated for responsiveness at high N).
        decim = 2 if int(n) >= 512 else 1
        phi_view = phi_shm[::decim, ::decim, ::decim]
        view_shape = phi_view.shape
        view_buf = np.empty(view_shape, dtype=np.float32)

        # Point-cloud parameters (voxel-centre dots).
        # We render the top-K brightest voxels in log1p-space each frame.
        # This is stable and avoids the "single hopping dot" effect.
        topk_points = 20000

        try:
            # VTK expects point dims; we are showing cell data.
            grid = ImageData()
            grid.dimensions = np.array(view_shape) + 1
            grid.spacing = (1.0, 1.0, 1.0)

            # Initial bind (log-compressed view to handle huge dynamic range)
            np.log1p(np.abs(phi_view), out=view_buf)
            grid.cell_data["phi"] = view_buf.ravel(order="F")

            # Robust initial range on the view buffer.
            # Use non-zero voxels only; the field is often extremely sparse and a single hot voxel
            # can crush the useful dynamic range.
            try:
                nz = view_buf[view_buf > 0.0]
                if nz.size >= 32:
                    lo = float(np.percentile(nz, 10.0))
                    hi = float(np.percentile(nz, 99.5))
                else:
                    lo = float(np.percentile(view_buf, 1.0))
                    hi = float(np.percentile(view_buf, 99.0))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    raise ValueError("bad clim")
            except Exception:
                lo = float(np.min(view_buf))
                hi = float(np.max(view_buf))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = 0.0, 1.0

            plotter = pv.Plotter(window_size=[1200, 900])
            plotter.title = "CAELIX Live"
            # Ensure normal mouse interaction (zoom/pan/rotate) across backends.
            try:
                plotter.enable_trackball_style()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                plotter.set_background((0.12, 0.12, 0.12))  # type: ignore[attr-defined]
            except Exception:
                pass

            # --- Actors ---
            # Volume actor (optional). Disabled by default; can be toggled with 'v'.
            vol = plotter.add_volume(
                grid,
                scalars="phi",
                opacity="linear",
                cmap="magma",
                shade=False,
                show_scalar_bar=True,
                clim=(lo, hi),
            )

            # Reduce smoothing in volume mode.
            try:
                prop = vol.GetProperty()  # type: ignore[attr-defined]
                try:
                    prop.SetInterpolationTypeToNearest()  # type: ignore[attr-defined]
                except Exception:
                    try:
                        prop.SetInterpolationType(0)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                pass

            # Point-cloud actor (default): coloured dots at voxel centres.
            pts_poly = pv.PolyData(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
            pts_poly["phi"] = np.array([0.0], dtype=np.float32)
            # Ensure there is a vertex cell for each point so VTK actually renders them.
            pts_poly.verts = np.array([1, 0], dtype=np.int64)
            pts_actor = plotter.add_mesh(
                pts_poly,
                scalars="phi",
                cmap="magma",
                render_points_as_spheres=True,
                point_size=6,
                opacity=0.95,
                show_scalar_bar=True,
            )
            # Keep points visible: initialise mapper range.
            try:
                pts_actor.mapper.scalar_range = (lo, hi)  # type: ignore[attr-defined]
            except Exception:
                pass

            try:
                plotter.add_outline()  # type: ignore[attr-defined]
                plotter.add_axes()  # type: ignore[attr-defined]
            except Exception:
                pass

            # Camera framing: explicitly frame the lattice cube so points/volume are in view
            # regardless of actor visibility quirks.
            try:
                L = float(n)
                # Bounds: [0, L] in each axis (points are in full-lattice coordinates).
                cx = 0.5 * L
                cy = 0.5 * L
                cz = 0.5 * L
                # Pull back along a diagonal.
                dist = 1.8 * L
                plotter.set_focus((cx, cy, cz))  # type: ignore[attr-defined]
                plotter.set_position((cx + dist, cy + dist, cz + dist))  # type: ignore[attr-defined]
                plotter.camera.up = (0.0, 0.0, 1.0)  # type: ignore[attr-defined]
                try:
                    plotter.reset_camera_clipping_range()  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception:
                # Fall back to PyVista's heuristic framing.
                try:
                    plotter.reset_camera()  # type: ignore[attr-defined]
                except Exception:
                    pass

            # View mode: default to point-cloud. Volume mode is allowed only for smaller N.
            allow_volume = int(n) < 512
            view_mode = {"mode": "points"}  # or "volume"

            # Start in points mode.
            try:
                if hasattr(vol, "SetVisibility"):
                    vol.SetVisibility(False)  # type: ignore[attr-defined]
                else:
                    vol.visibility = False  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                pts_actor.visibility = True  # type: ignore[attr-defined]
            except Exception:
                try:
                    pts_actor.SetVisibility(True)  # type: ignore[attr-defined]
                except Exception:
                    pass

            def _toggle_view() -> None:
                if not allow_volume:
                    return
                if view_mode["mode"] == "points":
                    view_mode["mode"] = "volume"
                    try:
                        if hasattr(vol, "SetVisibility"):
                            vol.SetVisibility(True)  # type: ignore[attr-defined]
                        else:
                            vol.visibility = True  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        pts_actor.visibility = False  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            pts_actor.SetVisibility(False)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                else:
                    view_mode["mode"] = "points"
                    try:
                        if hasattr(vol, "SetVisibility"):
                            vol.SetVisibility(False)  # type: ignore[attr-defined]
                        else:
                            vol.visibility = False  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        pts_actor.visibility = True  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            pts_actor.SetVisibility(True)  # type: ignore[attr-defined]
                        except Exception:
                            pass

            try:
                plotter.add_key_event("v", _toggle_view)  # type: ignore[attr-defined]
            except Exception:
                pass

            txt = plotter.add_text("", position="upper_left", font_size=10, color="white")
            # Some PyVista/VTK builds ignore the `color=` kwarg; force white via the VTK text property.
            try:
                if hasattr(txt, "GetTextProperty"):
                    tp = txt.GetTextProperty()  # type: ignore[attr-defined]
                    try:
                        tp.SetColor(1.0, 1.0, 1.0)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                pass

            def _set_hud(s: str) -> None:
                try:
                    if hasattr(txt, "SetText"):
                        txt.SetText(0, s)  # type: ignore[attr-defined]
                    elif hasattr(txt, "SetInput"):
                        txt.SetInput(s)  # type: ignore[attr-defined]
                except Exception:
                    pass

            _set_hud(
                f"mode=points  decim={decim}  max(phi_view)={float(np.max(phi_view)):.4f}  clim=({lo:.4g},{hi:.4g})"
            )

            interval_ms = max(1, int(1000 / max(1, int(fps))))

            # Auto-scale state: warm up, then slowly adapt.
            _clim_lo = float(lo)
            _clim_hi = float(hi)
            _clim_init_t = time.time()
            _clim_last_t = 0.0
            _clim_warmup_s = 2.0
            _clim_period_s = 0.5
            _clim_ema = 0.15

            def _tick() -> None:
                # Pull latest frame into the view buffer (log-compressed).
                try:
                    np.log1p(np.abs(phi_view), out=view_buf)
                    np.copyto(grid.cell_data["phi"], view_buf.ravel(order="F"), casting="no")
                    try:
                        grid.modified()
                    except Exception:
                        pass
                except Exception:
                    return

                try:
                    # Refresh clim occasionally, but stabilise it to avoid the "shrinking" illusion
                    # when a single hot voxel spikes.
                    tnow = time.time()
                    nonlocal _clim_lo, _clim_hi, _clim_last_t

                    if (tnow - _clim_last_t) >= _clim_period_s:
                        _clim_last_t = tnow

                        nz2 = view_buf[view_buf > 0.0]
                        if nz2.size >= 32:
                            p_lo = float(np.percentile(nz2, 10.0))
                            p_hi = float(np.percentile(nz2, 99.5))
                        else:
                            p_lo = float(np.percentile(view_buf, 1.0))
                            p_hi = float(np.percentile(view_buf, 99.0))

                        if np.isfinite(p_lo) and np.isfinite(p_hi) and p_hi > p_lo:
                            if (tnow - _clim_init_t) < _clim_warmup_s:
                                # During warmup, lock to the most permissive range seen so far.
                                _clim_lo = min(_clim_lo, p_lo)
                                _clim_hi = max(_clim_hi, p_hi)
                            else:
                                # After warmup, adapt slowly (EMA) so we don't chase spikes.
                                _clim_lo = (1.0 - _clim_ema) * _clim_lo + _clim_ema * p_lo
                                _clim_hi = (1.0 - _clim_ema) * _clim_hi + _clim_ema * p_hi

                            # Keep a sane separation.
                            if _clim_hi <= (_clim_lo + 1e-6):
                                _clim_hi = _clim_lo + 1e-3

                            try:
                                vol.mapper.scalar_range = (_clim_lo, _clim_hi)  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            try:
                                pts_actor.mapper.scalar_range = (_clim_lo, _clim_hi)  # type: ignore[attr-defined]
                            except Exception:
                                pass

                    lo3, hi3 = _clim_lo, _clim_hi

                    # Update point-cloud (default): render the top-K brightest voxels in log1p-space.
                    if view_mode["mode"] == "points":
                        try:
                            flat = view_buf.ravel(order="C")
                            vmax = float(np.max(flat))
                        except Exception:
                            vmax = 0.0
                            flat = None  # type: ignore[assignment]

                        if flat is None or (not np.isfinite(vmax)) or vmax <= 0.0:
                            pts_poly.points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                            pts_poly["phi"] = np.array([0.0], dtype=np.float32)
                        else:
                            k = int(topk_points)
                            if k <= 0:
                                k = 1
                            if k > flat.size:
                                k = int(flat.size)

                            # Argpartition is O(N) and much faster than a full sort.
                            try:
                                idx_flat = np.argpartition(flat, -k)[-k:]
                            except Exception:
                                idx_flat = np.arange(flat.size, dtype=np.int64)

                            # Drop zeros (common early on) to avoid a fake uniform cube of points.
                            try:
                                vals_k = flat[idx_flat]
                                keep = vals_k > 0.0
                                idx_flat = idx_flat[keep]
                                vals_k = vals_k[keep]
                            except Exception:
                                vals_k = None  # type: ignore[assignment]

                            if idx_flat.size == 0 or vals_k is None:
                                pts_poly.points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                                pts_poly["phi"] = np.array([0.0], dtype=np.float32)
                            else:
                                # Optionally thin if still too many points after dropping zeros.
                                if idx_flat.size > k:
                                    step = max(1, idx_flat.size // k)
                                    idx_flat = idx_flat[::step]
                                    vals_k = vals_k[::step]

                                # Convert flat indices -> 3D indices.
                                ii, jj, kk = np.unravel_index(idx_flat, view_shape)
                                pts = (np.stack([ii, jj, kk], axis=1).astype(np.float32) + 0.5) * float(decim)

                                pts_poly.points = pts
                                pts_poly["phi"] = vals_k.astype(np.float32, copy=False)

                        # Refresh vertex cells so all points render.
                        try:
                            npp = int(getattr(pts_poly, "n_points", 0))
                            if npp <= 0:
                                npp = 1
                            verts = np.empty((npp, 2), dtype=np.int64)
                            verts[:, 0] = 1
                            verts[:, 1] = np.arange(npp, dtype=np.int64)
                            pts_poly.verts = verts.ravel()
                        except Exception:
                            pass

                        try:
                            pts_poly.Modified()  # type: ignore[attr-defined]
                        except Exception:
                            pass

                    m = float(np.max(phi_view))
                    if not hasattr(_tick, "_m_prev"):
                        setattr(_tick, "_m_prev", m)
                    m_prev = float(getattr(_tick, "_m_prev"))
                    dm = m - m_prev
                    setattr(_tick, "_m_prev", m)

                    n_pts = 0
                    try:
                        n_pts = int(getattr(pts_poly, "n_points", 0))
                    except Exception:
                        n_pts = 0
                    _set_hud(
                        f"mode={view_mode['mode']}  decim={decim}  topk={topk_points}  max(phi_view)={m:.4f}  Δmax={dm:+.2e}  clim=({lo3:.4g},{hi3:.4g})  n_pts={n_pts}"
                    )
                except Exception:
                    pass

            # Force a first frame before the GUI loop starts.
            try:
                _tick()
            except Exception:
                pass

            # Drive updates manually for maximum compatibility across PyVista/VTK versions.
            # Some builds keep `app_running` False; `closed` is more reliable.
            try:
                _lv_log(log_path, "[liveview] loop=manual interactive_update=True")
                plotter.show(auto_close=False, interactive_update=True)
                while not getattr(plotter, "closed", False):  # type: ignore[attr-defined]
                    try:
                        _tick()
                    except Exception as e:
                        _lv_log(log_path, f"[liveview] tick error: {type(e).__name__}: {e}")
                    try:
                        # update() exists on many builds; fall back to render() otherwise.
                        if hasattr(plotter, "update"):
                            plotter.update()  # type: ignore[attr-defined]
                        else:
                            plotter.render()
                    except Exception as e:
                        _lv_log(log_path, f"[liveview] render error: {type(e).__name__}: {e}")
                    time.sleep(max(0.001, interval_ms / 1000.0))
            except Exception as e:
                _lv_log(log_path, f"[liveview] loop exception: {type(e).__name__}: {e}")
                _lv_log(log_path, traceback.format_exc())
                raise
            finally:
                try:
                    plotter.close()
                except Exception:
                    pass

        except Exception as e:
            _lv_log(log_path, f"[liveview] exception: {e}")
            _lv_log(log_path, traceback.format_exc())
            raise

    finally:
        try:
            shm.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--shm-name", type=str, default="CAELIX_shm")
    ap.add_argument("--connect-s", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--log-path", type=str, default="_Output/_liveview.log")
    args = ap.parse_args()

    lp = str(args.log_path).strip()
    run_viewer(
        int(args.n),
        shm_name=str(args.shm_name),
        connect_s=float(args.connect_s),
        fps=int(args.fps),
        log_path=(lp if lp != "" else None),
    )


if __name__ == "__main__":
    main()