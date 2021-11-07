import argparse

import gym
import h5py
import numpy as np

from pointmaze import maze_model


def generate_starts_and_targets(env_name, n_traj):
    env = gym.make(env_name)
    maze = env.str_maze_spec
    env = maze_model.MazeEnv(maze)

    starts_and_targets = []
    for _ in range(n_traj):
        s = env.reset()
        start = s[0:2]
        env.set_target()
        target = env._target
        starts_and_targets.append([start, target])
    starts_and_targets = np.array(starts_and_targets)

    fname = f'st-{env_name}-{n_traj}.hdf5'
    with h5py.File(fname, "w") as f:
        f.create_dataset("starts_and_targets", data=starts_and_targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='maze2d-large-dense-v1', help='Maze type')
    parser.add_argument('--n_traj', type=int, default=10, help='Number of trajectories to collect')
    args = parser.parse_args()

    generate_starts_and_targets(args.env_name, args.n_traj)


if __name__ == "__main__":
    main()
