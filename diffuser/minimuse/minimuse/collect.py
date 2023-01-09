import argparse
import diffuser.minimuse.minimuse.envs
import re
import gym
import sys
import numpy as np
import pickle as pkl
import torch.multiprocessing as mp
import torch

from copy import deepcopy
from tqdm import tqdm
from pathlib import Path


def filter_state(state):
    filtered_state = dict()
    # process tool state information
    state_keys = list(state.keys())
    # filter parts out of state
    filter_keywords = [
        "rgb",
        "info",
    ]
    pattern = "|".join(f"{k}" for k in filter_keywords)
    pattern = re.compile(pattern)

    for k in state_keys:
        if not pattern.match(k):
            filtered_state[k] = state[k]
    return filtered_state


def compute_workers_seed(episodes, num_workers, initial_seed):
    seeds_worker = [(initial_seed, episodes + initial_seed)]

    if num_workers > 0:
        episodes_per_worker = episodes // num_workers
        seeds_worker = np.arange(
            initial_seed, initial_seed + episodes, episodes_per_worker
        ).tolist()
        if len(seeds_worker) == num_workers + 1:
            seeds_worker[-2] = seeds_worker[-1]
            seeds_worker = seeds_worker[:-1]
        # the last worker handles the episodes outside of the last chunk
        seeds_worker.append(initial_seed + episodes)
        # transform (i0, i1, i2, ...) in ((i0, i1), (i1, i2), ...)
        for i, _ in enumerate(seeds_worker[:-1]):
            seeds_worker[i] = (seeds_worker[i], seeds_worker[i + 1])
        seeds_worker = seeds_worker[:-1]
        assert len(seeds_worker) == num_workers

    return seeds_worker


def get_args_parser():
    parser = argparse.ArgumentParser("Minimuse data collector script", add_help=False)
    parser.add_argument(
        "--env-name",
        default="Push-v0",
        type=str,
        help="Name of the environment",
    )
    parser.add_argument(
        "--episodes",
        default=1000,
        type=int,
        help="Number of expert demonstrations episodes to collect.",
    )
    parser.add_argument("--num-workers", default=20, type=int, help="Number of workers")
    parser.add_argument("--seed", default=0, type=int, help="Initial seed")
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory where the expert demonstrations will be saved",
    )
    return parser


def collect(worker_id, env_name, output_path, seeds, data_queue):
    torch.set_num_threads(1)
    env = gym.make(
        env_name,
    )

    pbar = None
    if worker_id == 0:
        pbar = tqdm(total=seeds[1] - seeds[0], ncols=80)

    stats = {
        "num_steps": [],
        "successful_seeds": [],
        "failure_seeds": [],
        "action_space": env.unwrapped.action_space.spaces,
        "cam_list": env.unwrapped.cam_list,
        "actions": [],
        "obs": [],
    }

    for seed in range(*seeds):
        episode_traj = []
        env.seed(seed)

        obs = env.reset()
        agent = env.unwrapped.oracle()

        for i in range(env.spec.max_episode_steps):
            action = agent.get_action(obs)
            if action is None:
                # If oracle is not able to solve the task
                info = {"success": False}
                stats["failure_seeds"].append(seed)
                break

            episode_traj.append((obs, action))
            obs, reward, done, info = env.step(action)
            if done:
                break

        if pbar is not None:
            pbar.update()

        # if trajectory is failed filter do not record in dataset
        if not info["success"]:
            print(seed)
            continue

        for i, step in enumerate(episode_traj):
            with open(str(output_path / f"{seed:08d}_{i:05d}.pkl"), "wb") as f:
                pkl.dump(step, f)

        for obs, action in episode_traj:
            state = filter_state(obs)
            stats["obs"] += [state]
            stats["actions"] += [action]
        stats["successful_seeds"].append(seed)
        stats["num_steps"].append(len(episode_traj))

    if pbar is not None:
        pbar.close()

    data_queue.put((worker_id, stats))
    print(f"Worker {worker_id} finished")
    del env


def main(args):
    output_path = Path(args.output_dir)
    initial_seed = args.seed
    episodes = args.episodes
    num_workers = args.num_workers

    if num_workers > episodes:
        num_workers = episodes

    seeds_worker = compute_workers_seed(episodes, num_workers, initial_seed)

    workers = []
    data_queue = mp.Queue()
    if num_workers == 0:
        collect(
            0,
            args.env_name,
            output_path,
            seeds_worker[0],
            data_queue,
        )

        i, collect_stats = data_queue.get()
    else:
        for i in range(num_workers):
            w = mp.Process(
                target=collect,
                args=(
                    i,
                    args.env_name,
                    output_path,
                    seeds_worker[i],
                    data_queue,
                ),
            )
            w.daemon = True
            w.start()
            workers.append(w)
        counter = 0
        data = []
        collect_stats = None
        while counter < num_workers:
            i, worker_stats = data_queue.get()

            if collect_stats:
                for k, v in worker_stats.items():
                    if k not in ["cam_list", "action_space"]:
                        collect_stats[k] += v
            else:
                collect_stats = deepcopy(worker_stats)
            print(f"Worker {i} data received")
            counter += 1

    # Process Statistics
    total_num_steps = sum(collect_stats["num_steps"])
    num_success_traj = len(collect_stats["successful_seeds"])
    failure_seeds = collect_stats["failure_seeds"]

    print(f"[Data Collection] - Number of successful trajectories: {num_success_traj}")
    print(f"[Data Collection] - Number of steps: {total_num_steps}")
    print(f"[Data Collection] - Failure seeds: {failure_seeds}")

    data = {}
    print("[Data Statistics] - Computing dataset statistics")
    all_actions = collect_stats.pop("actions")
    for action in all_actions:
        for k, v in action.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    all_obs = collect_stats.pop("obs")
    for obs in all_obs:
        for k, v in obs.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    velocity_dim = 0

    for k, v in collect_stats["action_space"].items():
        velocity_dim += v.shape[0]

    print(f"[Data Statistics] - Final dataset size {len(all_actions)}")
    stats = {
        "cam_list": collect_stats.pop("cam_list"),
        "action_space": collect_stats.pop("action_space"),
        "vel_dim": velocity_dim,
        "collect_stats": collect_stats,
        "dataset_size": len(all_actions),
        "traj_stats": {},
    }

    for k, v in data.items():
        stats["traj_stats"][k] = {
            "mean": np.mean(v, axis=0),
            "std": np.std(v, axis=0),
        }

    with open(str(output_path / "stats.pkl"), "wb") as f:
        pkl.dump(stats, f)

    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Minimuse data collector script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
