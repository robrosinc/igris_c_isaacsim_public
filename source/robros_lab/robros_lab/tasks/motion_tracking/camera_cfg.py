"""Calibrated OpenCV pinhole camera helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import isaaclab.sim as sim_utils


_OPENCV_PINHOLE_SCHEMA = "OmniLensDistortionOpenCvPinholeAPI"
_OPENCV_PINHOLE_COEFFICIENTS = (
    "k1",
    "k2",
    "p1",
    "p2",
    "k3",
    "k4",
    "k5",
    "k6",
    "s1",
    "s2",
    "s3",
    "s4",
)


def _apply_opencv_pinhole_distortion(
    prim,
    intrinsic_matrix: Sequence[Sequence[float]],
    distortion: Sequence[float],
    width: int,
    height: int,
) -> None:
    """Apply an OpenCV pinhole calibration to a spawned USD camera prim."""

    import omni.usd.schema.omni_lens_distortion as lens_distortion_schema
    from pxr import Gf, Plug, Usd

    Plug.Registry().RegisterPlugins(
        str(Path(lens_distortion_schema.__file__).resolve().parents[4] / "usd_plugins")
    )
    if Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(_OPENCV_PINHOLE_SCHEMA) is None:
        raise RuntimeError("Isaac Sim OpenCV pinhole lens-distortion schema is not registered.")

    coefficients = [float(value) for value in distortion]
    coefficients.extend([0.0] * (len(_OPENCV_PINHOLE_COEFFICIENTS) - len(coefficients)))
    fx, fy = float(intrinsic_matrix[0][0]), float(intrinsic_matrix[1][1])
    cx, cy = float(intrinsic_matrix[0][2]), float(intrinsic_matrix[1][2])

    prim.ApplyAPI(_OPENCV_PINHOLE_SCHEMA)
    prim.GetAttribute("omni:lensdistortion:model").Set("opencvPinhole")
    prim.GetAttribute("omni:lensdistortion:opencvPinhole:imageSize").Set(Gf.Vec2i(width, height))
    prim.GetAttribute("omni:lensdistortion:opencvPinhole:cx").Set(cx)
    prim.GetAttribute("omni:lensdistortion:opencvPinhole:cy").Set(cy)
    prim.GetAttribute("omni:lensdistortion:opencvPinhole:fx").Set(fx)
    prim.GetAttribute("omni:lensdistortion:opencvPinhole:fy").Set(fy)
    for name, value in zip(_OPENCV_PINHOLE_COEFFICIENTS, coefficients):
        prim.GetAttribute(f"omni:lensdistortion:opencvPinhole:{name}").Set(value)


def _opencv_pinhole_camera_spawner(
    intrinsic_matrix: Sequence[Sequence[float]],
    distortion: Sequence[float],
    width: int,
    height: int,
):
    @sim_utils.clone
    def spawn_camera(prim_path: str, cfg: sim_utils.PinholeCameraCfg, *args, **kwargs):
        prim = sim_utils.spawn_camera.__wrapped__(prim_path, cfg, *args, **kwargs)
        _apply_opencv_pinhole_distortion(prim, intrinsic_matrix, distortion, width, height)
        return prim

    return spawn_camera


def calibrated_pinhole_camera_cfg(
    intrinsic_matrix: Sequence[Sequence[float]],
    distortion: Sequence[float],
    width: int,
    height: int,
    clipping_range: tuple[float, float],
) -> sim_utils.PinholeCameraCfg:
    """Build the native OpenCV pinhole camera configuration used by the H1 tasks."""

    cfg = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[float(value) for row in intrinsic_matrix for value in row],
        width=width,
        height=height,
        focal_length=24.0,
        clipping_range=clipping_range,
    )
    if any(abs(float(value)) > 1.0e-12 for value in distortion):
        cfg.func = _opencv_pinhole_camera_spawner(
            intrinsic_matrix,
            distortion,
            width,
            height,
        )
    return cfg
