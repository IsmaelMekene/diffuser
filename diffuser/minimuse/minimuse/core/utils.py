import numpy as np
import mujoco_py
from pathlib import Path
from scipy.spatial.transform import Rotation


def assets_dir():
    return Path(__file__).parent.parent / "assets"


def quat_to_euler(quat, degrees):
    rotation = Rotation.from_quat(quat)
    return rotation.as_euler("xyz", degrees=degrees)


def muj_to_std_quat(muj_quat):
    quat = muj_quat.copy()
    quat[3] = muj_quat[0]
    quat[:3] = muj_quat[1:]
    return quat


def euler_to_quat(euler, degrees):
    rotation = Rotation.from_euler("xyz", euler, degrees=degrees)
    return rotation.as_quat()


def std_to_muj_quat(quat):
    muj_quat = quat.copy()
    muj_quat[0] = quat[3]
    muj_quat[1:] = quat[:3]
    return muj_quat


def xmat_to_std_quat(xmat):
    rotation = Rotation.from_matrix(xmat)
    return rotation.as_quat()


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation."""
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sim.forward()


def mul_quat(quat0, quat1):
    rot0 = Rotation.from_quat(quat0)
    rot1 = Rotation.from_quat(quat1)
    rot = rot0 * rot1
    return rot.as_quat()


def inv_quat(quat):
    rotation = Rotation.from_quat(quat)
    inv_rotation = rotation.inv()
    return inv_rotation.as_quat()


def get_axis_from_quat(quat, undefined=np.zeros(3)):
    tolerance = 1e-17
    vector = quat[0:3]
    norm = np.linalg.norm(vector)
    if norm < tolerance:
        # infinite set of possible axes
        return undefined
    else:
        return vector / norm


def get_angle_from_quat(quat):
    vector = quat[0:3]
    scalar = quat[-1]
    norm = np.linalg.norm(vector)
    return wrap_angle(2.0 * atan2(norm, scalar))
