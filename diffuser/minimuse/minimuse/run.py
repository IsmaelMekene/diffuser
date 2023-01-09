import argparse
from pathlib import Path
import numpy as np
import time

import diffuser.minimuse.minimuse.envs
from diffuser.minimuse.minimuse.core import constants
import gym

import matplotlib.pyplot as plt
from PIL import Image

import skvideo.io


def get_args_parser():
    parser = argparse.ArgumentParser("Run MiniMUSE environments", add_help=False)
    parser.add_argument("--env", default="Push-v0", type=str)
    parser.add_argument("--render", dest="render", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--video-dir", default="/tmp/demos", type=str)
    parser.add_argument("--episodes", default=100, type=int)
    parser.set_defaults(viewer=True, render=False)
    return parser


def main(args):
    if args.render:
        args.viewer = False

    env = gym.make(
        args.env,
        viewer=args.viewer,
        cam_render=args.render,
    )
    env.seed(args.seed)

    t0 = time.time()
    obs = env.reset()
    if args.viewer:
        env.unwrapped.render()
    if args.render:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        video_writer = skvideo.io.FFmpegWriter(str(video_dir / f"{args.seed}.mp4"))

    agent = env.unwrapped.oracle()
    actions = []
    seed = args.seed
    seed_max = args.seed + args.episodes
    done = False
    while True:
        action = agent.get_action(obs)
        if action is None or done:
            if info["success"]:
                text = "success"
            else:
                text = "failure"

            print(f"{seed}: {text} - {time.time()-t0} ")
            seed += 1
            if seed >= seed_max:
                break
            if args.render:
                video_writer.close()
                video_writer = skvideo.io.FFmpegWriter(str(video_dir / f"{seed}.mp4"))
            env.seed(seed)
            obs = env.reset()
            agent = env.unwrapped.oracle()
            action = agent.get_action(obs)

        actions.append(np.hstack([v for v in action.values()]))
        obs, reward, done, info = env.step(action)

        if args.viewer:
            env.unwrapped.render()

        if args.render:
            im = np.hstack(
                [[obs[f"rgb_{cam_name}"]] for cam_name in env.unwrapped.cam_list]
            )
            video_writer.writeFrame(im)

    actions = np.stack(actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run MUSE environments", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
