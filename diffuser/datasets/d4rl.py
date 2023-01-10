import os
import collections
import numpy as np
import gym
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    
    #for env in gym.envs.registry.env_specs:
    #    if 'Push-v0' in env:
    #      print('\nRemove {} from registry\n'.format(env))
    #     del gym.registry.env_specs[env]
    env = wrapped_env.unwrapped
    if 'push-v0' in str(env).lower():
      wrapped_env._max_episode_steps = 600
      wrapped_env.reward_threshold=1.0
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    dataset = env.get_dataset()

    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    return dataset

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
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
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


def sequence_dataset_pushv0(env_name, data_dir,
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

    N = len(entries)
    i=0
    data_dict = { "observations": [],
                "actions": [],
                "rewards": [],
                "terminals": []
                }
    for h5file in entries: # in every h5file is stored an episode 
        h5path = os.path.join(data_dir, h5file)
        with h5py.File(h5path, 'r') as dataset_file:
            episode_data = {}
            episode_length = min(max_path_length, len(dataset_file['rewards'])
            epi_final_length = 
            episode_data['actions'] = dataset_file[1]['actions'][:episode_length]
            episode_data['observations'] = dataset_file[2]['observations'][:episode_length]
            episode_data['rewards'] = dataset_file[2]['rewards'][:episode_length]
            if obs_type == 'state_fetures':
                episode_data['observations'] = dataset_file[0][:episode_length]
            else: # obs_type='pixel': the observation are raw pixel images
                episode_data['observations'] = dataset_file[0][:episode_length]
                
            #for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            #try:  # first try loading as an array
            #    data_dict[k] = np.concatenate((data_dict[k], dataset_file[k][:]), axis=0)
            #except ValueError as e:  # try loading as a scalar
            #    data_dict[k] = dataset_file[k][()]
        i+=1
        if i>= N:
            break

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
