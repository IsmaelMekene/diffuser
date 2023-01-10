from collections import namedtuple
import numpy as np
import torch
import pdb
import h5py
import torchvision.transforms as transforms

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset, sequence_dataset_pushv0
from .normalization import DatasetNormalizer, DatasetNormalizerPush
from .buffer import ReplayBuffer
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
        
        
        itr = sequence_dataset_pushv0(data_dir,preprocess_fn=self.preprocess_fn,
                        obs_type=obs_type, max_path_length=max_path_length)

        #itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
           fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizerPush(fields, normalizer, path_lengths=fields['path_lengths'])
        
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = next(itr)['observations'][0].shape #fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = len(os.listdir(data_dir)) #fields.n_episodes #
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

       
    def process_img(img):
        return transform_img(img).numpy()

    def sequence_dataset_pushv0(data_dir,
        preprocess_fn, obs_type='state_features', max_path_length=600):
        """
        Returns an iterator through trajectories.
        Args:
            env: An OfflineEnv object.
            data_dir: data directory
            max_path_length: max number of steps per episode
            dataset: An optional dataset to pass in for processing. If None,
                the dataset will default to env.get_dataset()
            **kwargs: Arguments to pass to env.get_dataset().
        Returns:
            An iterator through dictionaries with keys:
                observations
                actions
                rewards
                terminals
        """
        entries = os.listdir(data_dir)
        #dataset = get_dataset(env)
        #dataset = preprocess_fn(dataset)

        for h5file in entries: # in every h5file is stored an episode 
            h5path = os.path.join(data_dir, h5file)
            with h5py.File(h5path, 'r') as dataset_file:
                episode_data = {}
                episode_length = min(max_path_length, len(dataset_file['rewards']))
                episode_data['actions'] = dataset_file[1]['actions'][:episode_length]
                episode_data['terminals'] = dataset_file[2]['terminals'][:episode_length]
                episode_data['rewards'] = dataset_file[2]['rewards'][:episode_length]
                if obs_type == 'state_fetures':
                    episode_data['observations'] = dataset_file[0][:episode_length]
                else: # obs_type='pixel': the observation are raw pixel images
                    episode_data['observations'] = process_img( dataset_file[0][:episode_length] )
                yield episode_data


    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        if 'push' in self.env_name.lower():
            keys = ['actions']
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        #if 'push' in self.env_name.lower():
        episode =  next(self.itr())
        observations = episode['observations']
        actions = episode['actions']
    
        """
        begin modified 
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch
        """
        ## begin modifed (added lines)
        batch = BatchV2(actions, observations[0]) # take only the first observation of the trajectory
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


class ValueDataset(SequenceDataset):
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
