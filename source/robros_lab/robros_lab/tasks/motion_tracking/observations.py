"""Observation configurations for IGRIS-C motion tracking inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robros_lab.tasks.base import mdp as base_mdp

from .mdp import reference_observations as reference_mdp
from .student_contract import STUDENT_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.managers import ObservationManager


STUDENT_HISTORY_LENGTH = 20

STUDENT_OBSERVATION_TERM_ORDER = (
    "joint_pos",
    "joint_vel",
    "projected_gravity",
    "base_ang_vel",
    "actions",
    "motion_joint_pos",
    "motion_joint_vel",
    "motion_anchor_pos_z",
    "motion_anchor_ori_b",
    "motion_body_pos_b",
    "motion_anchor_lin_vel_b",
    "motion_anchor_ang_vel_b",
)

STUDENT_HISTORY_TERM_NAMES = (
    "joint_pos",
    "joint_vel",
    "projected_gravity",
    "base_ang_vel",
    "actions",
)


@configclass
class StudentObservationsCfg(ObsGroup):
    """Exact student observation contract used by student_108.pt at inference."""

    # Proprioception history. Declaration order is part of the checkpoint contract.
    joint_pos = ObsTerm(
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(STUDENT_JOINT_NAMES),
                preserve_order=True,
            ),
        },
        func=base_mdp.joint_pos_feedback_rel,
        noise=None,
        history_length=STUDENT_HISTORY_LENGTH,
    )
    joint_vel = ObsTerm(
        func=base_mdp.finite_difference_obs,
        params={
            "obs_func": base_mdp.joint_pos_feedback_rel,
            "observed_asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(STUDENT_JOINT_NAMES),
                preserve_order=True,
            ),
            "enable_lpf": True,
            "cutoff_frequency_hz": 6.0,
        },
        history_length=STUDENT_HISTORY_LENGTH,
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=None,
        history_length=STUDENT_HISTORY_LENGTH,
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        noise=None,
        history_length=STUDENT_HISTORY_LENGTH,
    )
    actions = ObsTerm(
        func=mdp.last_action,
        history_length=STUDENT_HISTORY_LENGTH,
    )

    # Current motion reference supplied by the replaceable motion-tracking command.
    motion_joint_pos = ObsTerm(
        func=reference_mdp.motion_joint_pos,
        params={"command_name": "motion"},
        noise=None,
    )
    motion_joint_vel = ObsTerm(
        func=reference_mdp.motion_joint_vel,
        params={"command_name": "motion"},
        noise=None,
    )
    motion_anchor_pos_z = ObsTerm(
        func=reference_mdp.motion_anchor_pos_z,
        params={"command_name": "motion"},
        noise=None,
    )
    motion_anchor_ori_b = ObsTerm(
        func=reference_mdp.motion_anchor_ori_b,
        params={"command_name": "motion"},
        noise=None,
    )
    motion_body_pos_b = ObsTerm(
        func=reference_mdp.motion_body_pos_b,
        params={"command_name": "motion"},
        noise=None,
    )
    motion_anchor_lin_vel_b = ObsTerm(
        func=reference_mdp.motion_anchor_lin_vel_b,
        params={"command_name": "motion"},
        noise=None,
    )
    motion_anchor_ang_vel_b = ObsTerm(
        func=reference_mdp.motion_anchor_ang_vel_b,
        params={"command_name": "motion"},
        noise=None
    )

    def __post_init__(self) -> None:
        # Student inference must not apply training-time observation corruption.
        self.enable_corruption = False
        self.concatenate_terms = True

        for term_name in STUDENT_HISTORY_TERM_NAMES:
            term_cfg = getattr(self, term_name)
            if term_cfg.history_length != STUDENT_HISTORY_LENGTH:
                raise ValueError(
                    f"Student observation {term_name!r} must use history length "
                    f"{STUDENT_HISTORY_LENGTH}, got {term_cfg.history_length}."
                )


@configclass
class MotionTrackingObservationsCfg:
    """Student policy observations for motion inference."""

    student: StudentObservationsCfg = StudentObservationsCfg()


def validate_student_observation_manager(observation_manager: ObservationManager) -> None:
    """Validate the resolved student term order after environment construction."""

    active_terms = observation_manager.active_terms
    if "student" not in active_terms:
        raise ValueError(
            f"Observation manager does not expose a 'student' group. "
            f"Available groups: {tuple(active_terms)}."
        )

    actual_term_order = tuple(active_terms["student"])
    if actual_term_order != STUDENT_OBSERVATION_TERM_ORDER:
        raise ValueError(
            "Resolved student observation order does not match student_108.pt: "
            f"expected {STUDENT_OBSERVATION_TERM_ORDER}, got {actual_term_order}."
        )

    if not observation_manager.group_obs_concatenate["student"]:
        raise ValueError("The student observation group must concatenate its terms.")
