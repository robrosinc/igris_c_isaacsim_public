"""Calibrated OpenCV pinhole camera helpers."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils


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
        k1, k2, p1, p2, k3 = (float(value) for value in distortion)
        cfg.distortion = sim_utils.OpenCvPinholeDistortionCfg(
            fx=float(intrinsic_matrix[0][0]),
            fy=float(intrinsic_matrix[1][1]),
            cx=float(intrinsic_matrix[0][2]),
            cy=float(intrinsic_matrix[1][2]),
            image_size=(width, height),
            k1=k1,
            k2=k2,
            p1=p1,
            p2=p2,
            k3=k3,
        )
    return cfg
