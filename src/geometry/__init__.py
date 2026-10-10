"""Geometry subpackage for SO(3)^K operations, kinematics, and Riemannian optimization."""

from src.geometry.so3 import (
    hat,
    vee,
    so3_exp_map,
    so3_log_map,
    exp_map_at_point,
    log_map_at_point,
    project_to_so3,
    project_to_tangent_space,
    so3_geodesic_distance,
    so3_product_distance,
    sample_uniform_so3,
    trajectory_regularizer,
)
from src.geometry.radam import RiemannianAdam
from src.geometry.smpl_utils import (
    SMPL_JOINT_NAMES,
    BODY_JOINT_NAMES,
    SMPL_PARENTS,
    DifferentiableKinematics,
)

__all__ = [
    "hat",
    "vee",
    "so3_exp_map",
    "so3_log_map",
    "exp_map_at_point",
    "log_map_at_point",
    "project_to_so3",
    "project_to_tangent_space",
    "so3_geodesic_distance",
    "so3_product_distance",
    "sample_uniform_so3",
    "trajectory_regularizer",
    "RiemannianAdam",
    "SMPL_JOINT_NAMES",
    "BODY_JOINT_NAMES",
    "SMPL_PARENTS",
    "DifferentiableKinematics",
]
