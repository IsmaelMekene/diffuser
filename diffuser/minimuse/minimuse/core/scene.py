import numpy as np
import copy

from mujoco_py import load_model_from_path
from mujoco_py import MjSim, MjViewer

from diffuser.minimuse.minimuse.core import constants, utils


class Scene:
    def __init__(
        self,
        model_path="",
        viewer=False,
        tools_loaded=("broom",),
    ):
        if not model_path:
            model_path = utils.assets_dir() / "base_env.xml"
        else:
            model_path = utils.assets_dir() / model_path

        self.model = load_model_from_path(str(model_path))

        self.sim = MjSim(self.model)
        self._initial_state = copy.deepcopy(self.sim.get_state())
        self._tools_loaded = tools_loaded

        if viewer:
            self.viewer = MjViewer(self.sim)
            self._setup_viewer()
        else:
            self.viewer = None

        self.controller_dt = constants.CONTROLLER_DT
        self.max_tool_velocity = constants.MAX_TOOL_VELOCITY

        self._mocap_pos = dict()
        self._mocap_quat = dict()
        self._init_tool_mocap()

    def reset(self, mocap_pos, mocap_quat, workspace):
        self.sim.set_state(self._initial_state)

        self.total_steps = 0

        utils.reset_mocap_welds(self.sim)
        self._init_tool_mocap()

        self._mocap_pos.update(mocap_pos)
        if mocap_quat is not None:
            self._mocap_quat.update(mocap_quat)

        # initialize tool pose
        for name, target_pos in self._mocap_pos.items():
            target_quat = self._mocap_quat[name]
            self._set_mocap_target(name, target_pos, target_quat)

        self.sim.forward()

    def warmup(self):
        """
        warmup simulation by running simulation steps
        set mocap target and resolves penetrating objects

        WARNING: always call this function after setting all the model poses
        typically at the end of the reset of an environment before observe
        """
        for _ in range(constants.WARMUP_STEPS):
            self.sim.step()

    def step(self, action, workspace):
        for name_ee, action_ee in action.items():
            assert action_ee.shape[0] == 6
            # update the mocap pose and gripper target angles
            self._step_mocap(name_ee, action_ee[:3], action_ee[3:], workspace)
        # run simulation steps for the arms and gripper to match defined targets
        for i in range(constants.SIM_STEPS):
            self.sim.step()
            if self.viewer and self.total_steps % constants.RENDER_STEPS == 0:
                self.render()
            self.total_steps += 1

    def render(self):
        self.viewer.render()

    def render_camera(self, width, height, camera_name):
        im = self.sim.render(width=width, height=height, camera_name=camera_name)
        im = np.flip(im, 0)
        return im

    def get_site_pos(self, name):
        return np.copy(self.sim.data.get_site_xpos(name))

    def get_site_quat(self, name):
        return np.copy(utils.xmat_to_std_quat(self.sim.data.get_site_xmat(name)))

    def get_geom_pos(self, name):
        return np.copy(self.sim.data.get_geom_xpos(name))

    def get_geom_quat(self, name):
        return np.copy(
            utils.std_to_muj_quat(
                utils.xmat_to_std_quat(self.sim.data.get_geom_xmat(name))
            )
        )

    def get_body_pos(self, name):
        return np.copy(self.sim.data.get_body_xpos(name))

    def get_body_quat(self, name):
        return np.copy(utils.std_to_muj_quat(self.sim.data.get_body_xquat(name)))

    def get_joint_qpos(self, name):
        return np.copy(self.sim.data.get_joint_qpos(name))

    def get_ee_pos(self, name):
        return np.copy(self.sim.data.get_mocap_pos(f"{name}_mocap"))

    def get_ee_quat(self, name):
        return np.copy(self.sim.data.get_mocap_quat(f"{name}_mocap"))

    def get_geom_size(self, name):
        geom_id = self.model.geom_name2id(name)
        geom_size = self.model.geom_size[geom_id]
        return geom_size

    def set_joint_qpos(self, name, qpos):
        self.sim.data.set_joint_qpos(name, qpos)
        qvel = self.sim.data.get_joint_qvel(name)
        self.sim.data.set_joint_qvel(name, np.zeros_like(qvel))
        self.sim.forward()

    def _step_mocap(self, name, linvel, angvel, workspace):
        linvel = linvel * self.controller_dt
        linvel = np.clip(
            linvel, -constants.MAX_TOOL_VELOCITY[0], constants.MAX_TOOL_VELOCITY[0]
        )
        target_pos = self._mocap_pos[name] + linvel
        target_pos = np.clip(target_pos, workspace[0], workspace[1])

        angvel = angvel * self.controller_dt
        angvel = np.clip(
            angvel, -constants.MAX_TOOL_VELOCITY[1], constants.MAX_TOOL_VELOCITY[1]
        )
        mocap_quat = utils.muj_to_std_quat(self._mocap_quat[name])
        mocap_orn = utils.quat_to_euler(mocap_quat, False)
        target_orn = mocap_orn + angvel

        target_quat = utils.euler_to_quat(target_orn, False)
        target_quat = utils.std_to_muj_quat(target_quat)
        self._set_mocap_target(name, target_pos, target_quat)

    def _set_mocap_target(self, name, target_pos=None, target_quat=None):
        if target_pos is not None:
            self._mocap_pos[name] = target_pos
        if target_quat is not None:
            self._mocap_quat[name] = target_quat
        self.sim.data.set_mocap_pos(f"{name}_mocap", self._mocap_pos[name])
        self.sim.data.set_mocap_quat(f"{name}_mocap", self._mocap_quat[name])
        self.sim.forward()

    def _setup_viewer(self):
        body_id = self.sim.model.body_name2id(f"{self._tools_loaded[0]}_mocap")
        lookat = self.sim.data.body_xpos[body_id]
        viewer_cam = self.viewer.cam
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.6
        self.viewer.cam.azimuth = 0.0
        self.viewer.cam.elevation = -20.0

    def _init_tool_mocap(self):
        for tool_name in self._tools_loaded:
            if tool_name == "broom":
                tool_pos = [0, 0, 0.2]
                tool_quat = utils.euler_to_quat((0, 0, 0), degrees=True)
                tool_quat = utils.std_to_muj_quat(tool_quat)
            self._mocap_pos[tool_name] = tool_pos
            self._mocap_quat[tool_name] = tool_quat

        self.sim.forward()
