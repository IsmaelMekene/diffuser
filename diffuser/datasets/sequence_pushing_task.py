from collections import namedtuple
import numpy as np
import torch
import pdb
import h5py
import torchvision.transforms as transforms
import random 

from diffuser.datasets.preprocessing import get_preprocess_fn
from diffuser.datasets.d4rl import load_environment
#from diffuser.datasets.normalization import DatasetNormalizer, DatasetNormalizerPush
#from diffuser.datasets.buffer import ReplayBuffer
import os

Batch = namedtuple('Batch', 'trajectories conditions')

transform_img = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

## begin modified (added line)
BatchV2 = namedtuple('BatchV2', 'actions states')
## end modified

ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


class SequenceDatasetPush(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None,
        obs_type='state_features', data_dir='collected_data' ):
        self.obs_type = obs_type
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env_name = env
        self.obs_type = obs_type
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.data_dir = data_dir
        self.nb_loaded = 0

        entries = os.listdir(data_dir)
        random.shuffle(entries)
        i=0
        # remove file whivh does not have the .hdf5 extentension
        while i<len(entries):
          if entries[i][-5:] !='.hdf5':
            entries.pop(i)
            i-=1
          i+=1
        print(f'\n Pushing task number  of trajectories: {len(entries)}\n')
        self.entries = entries # every file in entries contains a path
        h5path = os.path.join(data_dir, entries[0])
        self.n_episodes = len(entries)
        with h5py.File(h5path, 'r') as dataset_file:
            if self.obs_type == 'state_features':
                self.observation_dim = dataset_file['observations_features'].shape
            else: # obs_type='pixel': the observation are raw pixel images
                self.observation_dim = dataset_file['observations'].shape #fields.observations.shape[-1]
            self.action_dim = dataset_file['actions'][0].shape[0]
  
    def process_img(self, img):
        return transform_img(img).numpy()

    def normalize(self, keys=['actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        if 'push' in self.env_name.lower():
            keys = ['actions']
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def normalize_actions(self, actions, path_start, path_end):
        mean_actions = 0.0 #np.mean(actions)
        std_actions = 0.015 #np.std(actions)
        #eps=1e-10
        return (actions[path_start:path_end]-mean_actions)/(std_actions)

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.entries)

    def safe_batch(self):
        return self.prev_batch
    
    def __getitem__(self, idx, eps=1e-4):
        #path_ind, start, end = self.indices[idx]
        h5file = self.entries[idx%self.n_episodes]
        h5path = os.path.join(self.data_dir, h5file)
        with h5py.File(h5path, 'r') as dataset_file:
            episode_length = len(dataset_file['rewards'])
            if episode_length < self.horizon:
                print("[Using the previous batch]...")
                return self.safe_batch()
            path_start = np.random.randint(0, episode_length-self.horizon+1)
            path_end = path_start + self.horizon
            #actions = self.normalize_actions(dataset_file['actions'], path_start, path_end)
            actions = dataset_file['actions'][path_start:path_end]/0.015
            if self.obs_type == 'state_features':
                first_observation = dataset_file['observations_features'][:]/0.01
            else: # obs_type='pixel': the observation are raw pixel images
                first_observation = self.process_img(np.array(dataset_file["observations"]))

        batch = BatchV2(actions, first_observation)
        self.prev_batch = batch
        return batch


class GoalDataset(SequenceDatasetPush):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDatasetPush):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
