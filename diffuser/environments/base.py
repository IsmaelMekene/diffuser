import numpy as np
import gym
from gym.utils import seeding

from diffuser.minimuse.minimuse.core.scene import Scene
from diffuser.minimuse.minimuse.core.utils import muj_to_std_quat, quat_to_euler
from diffuser.minimuse.minimuse.core import constants
from diffuser.minimuse.minimuse.agents.script import ScriptAgent
from diffuser.minimuse.minimuse.envs import utils


WORKSPACE = np.array([[-0.2, -0.3, 0.00], [0.2, 0.3, 0.2]])


class BaseEnv(gym.Env):
    def __init__(
        self,
        model_path="",
        viewer=False,
        cam_resolution=constants.RENDER_RESOLUTION,
        cam_render=True,
        # cam_list=["frontal_camera", "lateral_camera", "top_camera"],
        cam_list=["top_camera"],
        cam_crop=True,
        tool_name="broom",
    ):
        self.scene = Scene(model_path, viewer, tools_loaded=(tool_name,))

        self.cam_render = cam_render
        self.cam_list = cam_list
        self.cam_resolution = cam_resolution
        self.cam_crop = cam_crop

        # workspace
        self.workspace = WORKSPACE

        self.tool_name = tool_name
        self.tool_workspace = self.workspace.copy()

        self.obj_workspace = self.workspace + np.array(
            [[0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]]
        )

        self._np_random = np.random

    def seed(self, seed=None):
        np_random, seed = seeding.np_random(seed)
        self._np_random = np_random
        self._seed = seed
        return seed

    def reset(self, mocap_target_pos=None, mocap_target_quat=None):
        if mocap_target_pos is None:
            mocap_target_pos = self._np_random.uniform(
                self.tool_workspace[0], self.tool_workspace[1]
            )

        mocap_target_pos_dict = dict()
        mocap_target_pos_dict[self.tool_name] = mocap_target_pos
        if mocap_target_quat is None:
            mocap_target_quat_dict = None
        else:
            mocap_target_quat_dict = dict()
            mocap_target_quat_dict[self.tool_name] = mocap_target_quat

        self.scene.reset(
            mocap_pos=mocap_target_pos_dict,
            mocap_quat=mocap_target_quat_dict,
            workspace=self.workspace,
        )

    def render(self):
        self.scene.render()

    def observe(self):
        obs = dict()
        obs["tool_pos"] = self.scene.get_ee_pos(self.tool_name)
        obs["tool_quat"] = muj_to_std_quat(self.scene.get_ee_quat(self.tool_name))
        obs["tool_theta"] = quat_to_euler(obs["tool_quat"], False)[-1]

        if self.cam_render:
            w, h = self.cam_resolution
            for cam_name in self.cam_list:
                im = self.scene.render_camera(w, h, cam_name)
                if self.cam_crop:
                    im = utils.realsense_resize_crop(im, im_type="rgb")
                obs[f"rgb_{cam_name}"] = im
        return obs

    def step(self, action):
        action_np = np.zeros(6)
        action_np[:3] = action.get("linear_velocity", np.zeros(3))

        action_np[3:] = action.get("angular_velocity", np.zeros(3))

        action_dict = dict()
        action_dict[self.tool_name] = action_np

        self.scene.step(action_dict, self.workspace)
        obs = self.observe()
        success = self.is_success()
        reward = float(success)
        done = success
        info = dict(success=success)
        return obs, reward, done, info

    def oracle(self):
        return ScriptAgent(self)
