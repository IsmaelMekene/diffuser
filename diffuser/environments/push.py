import numpy as np

from copy import copy
from diffuser.minimuse.minimuse.core import constants
from diffuser.minimuse.minimuse.envs.base import BaseEnv
from diffuser.minimuse.minimuse.envs.utils import random_xy, create_action_space
from diffuser.minimuse.minimuse.oracles.multimodal_push import MultimodalPushOracle
from gym.envs.mujoco import mujoco_env
#import cv2
from gym import utils
import pickle
import os
from tqdm import tqdm
import h5py

class PushEnv(BaseEnv):
    def __init__(self, obs_type='pixel', **kwargs):
        super().__init__(model_path=os.path.join(
            os.path.dirname(__file__), 'assets/push_env.xml'), **kwargs)
        #asset_path = os.path.join(
        #    os.path.dirname(__file__), 'assets/push_env.xml')
        #mujoco_env.MujocoEnv.__init__(self, asset_path, 5)
        utils.EzPickle.__init__(self)
        self.obs_type= obs_type
        self.num_cubes = 2
        self.cubes_name = [f"cube{i}_joint" for i in range(self.num_cubes)]
        self.goals_name = ["marker0", "marker1"]

        self.ref_min_score = 5.
        self.ref_max_score = 50.

        self.scene.max_tool_velocity = constants.MAX_TOOL_VELOCITY

        workspace_x_center = (
            1 * (self.workspace[1][0] - self.workspace[0][0]) / 8
        ) + self.workspace[0][0]

        self.obj_workspace = self.workspace.copy()
        self.obj_workspace[0, 0] = workspace_x_center
        self.obj_workspace += np.array([[0.2, 0.08, 0.01], [-0.1, -0.08, -0.01]])

        self.goal_workspace = self.workspace.copy()
        self.goal_workspace[1, 0] = workspace_x_center
        self.goal_workspace += np.array([[0.08, 0.08, 0.01], [-0.08, -0.08, -0.01]])

        self.cubes_color = [(0.502, 0.769, 0.388), (0.502, 0.769, 0.388)]
        self.markers_color = [(0.592, 0.188, 0.365), (0.592, 0.188, 0.365)]

        self.default_tool_height = 0.065
        self.min_x_distance = 0.01
        self.min_y_distance = 0.01

        self._initial_tool_pos = np.array([self.workspace[1][0], 0.0, 0.0])
        self._initial_tool_pos[-1] = self.default_tool_height

        self.action_space = create_action_space(self.scene, "xy_linear_velocity")

    def reset(self):
        super().reset(mocap_target_pos=self._initial_tool_pos)

        valid = False

        while not valid:
            cubes_qpos = random_xy(
                self.num_cubes,
                self._np_random,
                self.obj_workspace,
                z_pos=0.03,
            )
            valid = True
            for i in range(1, self.num_cubes):
                valid = np.abs(cubes_qpos[i][1] - cubes_qpos[i - 1][1]) > 0.06
                if not valid:
                    break

        for i in range(self.num_cubes):
            self.scene.set_joint_qpos(self.cubes_name[i], cubes_qpos[i])

        self.scene.warmup()
        obs = self.observe()
        return obs

    def observe(self):
        obs = super().observe()
        for i in range(self.num_cubes):
            obs[f"cube{i}_pos"] = self.scene.get_site_pos(f"cube{i}")
            obs[f"cube{i}_quat"] = self.scene.get_body_quat(f"cube{i}")
        obs["goal0_pos"] = self.scene.get_site_pos("marker0")
        obs["goal1_pos"] = self.scene.get_site_pos("marker1")
        return obs

    def step(self, action):

        action = copy(action)
        linear_velocity = np.zeros(3)
        if type(action) == np.ndarray:
          linear_velocity[:2] = action
          action = {"xy_linear_velocity":linear_velocity
          }
          return super().step(action)
        linear_velocity[:2] = action.pop("xy_linear_velocity")
        action["linear_velocity"] = linear_velocity
        return super().step(action)

    def is_success(self):
        obj_success = [False] * self.num_cubes
        for i in range(self.num_cubes):
            cube_qpos = self.scene.get_joint_qpos(self.cubes_name[i])
            for goal_name in self.goals_name:
                goal_qpos = self.scene.get_site_pos(goal_name)
                success = (
                    abs(goal_qpos[0] - cube_qpos[0]) < self.min_x_distance
                    and abs(goal_qpos[1] - cube_qpos[1]) < self.min_y_distance
                )
                if success:
                    obj_success[i] = True
                    break
        return np.all(obj_success)

    def oracle(self):
        return MultimodalPushOracle(
            self._np_random, self.min_x_distance, self.min_y_distance
        )

    def get_keys(self, h5file):
      keys = []
      def visitor(name, item):
          if isinstance(item, h5py.Dataset):
              keys.append(name)

      h5file.visititems(visitor)
      return keys
    
    def get_dataset_pixel_observation(self, data_path, max_nb_entries=10):
        data_path = '/content/Drive/MyDrive/project_recvis_diffuser/diffuser_img/diffuser/collected_data_with_reward_img3'
        entries = os.listdir(data_path)
        N = min(len(entries), max_nb_entries)
        i=0
        data_dict = { "observations": [],
                    "actions": [],
                    "rewards": [],
                    "terminals": []
                    }
        for h5file in entries:  
          h5path = os.path.join(data_path, h5file)
          with h5py.File(h5path, 'r') as dataset_file:
            for k in tqdm(self.get_keys(dataset_file), desc="load datafile"):
              try:  # first try loading as an array
                data_dict[k] = np.concatenate((data_dict[k], dataset_file[k][:]), axis=0)
              except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
          i+=1
          if i>= N:
            break
        return data_dict

    def get_normalized_score(self, score):
        if (self.ref_max_score is None) or (self.ref_min_score is None):
            raise ValueError("Reference score not provided for env")
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)

    def get_dataset(self, dataset_path='/content/Drive/MyDrive/project_recvis_diffuser/minimuse/collected_data_with_reward2/', pixel=False, max_nb_entries=10):
        """
        Inputs:
        -pixel: If pixel is True, the state s_t are RGB images, otherwise s_t is represented by an
        array [tool_pos, tool_quat, tool_theta, cube0_pos, cube0_quat, cube1_pos, cube1_quat,
        goal0_pos, goal1_pos]
        -max_nb_entries: size of the data set (i.e. number of sequences of the type
        (state, action, reward, terminals))
        """
        data_path = 'collected_data_with_reward3'
        if self.obs_type=='pixel':
          return self.get_dataset_pixel_observation(data_path, max_nb_entries)
  
        
        entries = os.listdir(data_path)
        entries.sort()
        N = len(entries)
        if max_nb_entries is not None:
            N = min(len(entries), max_nb_entries) # size of the data set (i.e number of (actions, observations) pairs)
        dataset = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminals": []
        }
        load = pickle.load(open(os.path.join(data_path, entries[0]), 'rb'))
        
        # state
        if pixel:
            img0 = load[0]['rgb_top_camera'] # observation (image)
            dataset['observations'] = np.zeros( tuple([N])+img0.shape, dtype=int)
        else:
            load[0].pop('rgb_top_camera') # remove the image from the obsevations
            load[0]['tool_theta'] = np.array([load[0]['tool_theta']])
            obs = np.concatenate(list(load[0].values()))
            state_dim = len(obs)
            dataset['observations'] = np.zeros((N,state_dim))

        # action
        action_dim = len(load[1]['xy_linear_velocity'])
        dataset['actions'] = np.zeros((N, action_dim))

        # We added reward and terminal-sate indicator
        dataset['rewards'] = np.zeros(N)
        dataset['terminals'] = np.zeros(N, dtype=bool)
        i=0
        for pkl_file in entries:
            load = pickle.load(open(os.path.join(data_path, pkl_file), 'rb'))

            # state
            if pixel:
                dataset['observations'][i] = load[0]['rgb_top_camera']
            else:
                load[0].pop('rgb_top_camera') # remove the image from the obsevations
                load[0]['tool_theta'] = np.array([load[0]['tool_theta']])
                dataset['observations'][i]=np.concatenate(list(load[0].values()))
            # action, reward, terminal
            dataset['actions'][i] = load[1]['xy_linear_velocity']
            dataset['rewards'][i]=load[2]['reward']
            dataset['terminals'][i]= load[2]['terminal']
            i+=1
            #dataset['observations'].append(load[0]['rgb_top_camera']) # current observation (top view camera)
            #dataset['actions'].append(load[1]['xy_linear_velocity'])
            #dataset['rewards'].append(load[2]['reward'])
            #dataset['terminals'].append(load[2]['terminal'])
            if i>= N:
                break
            if i%(N//10)==0:
                print(100*i//N, " % of the pushing task dataset loaded")

        return dataset
  