import gym
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import pickle
import gzip
import h5py
import argparse

from pointmaze.waypoint_controller import NoisyWaypointController, RandomController, WaypointControllerWrapper


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            'infos/start': [],
            }

def append_data(data, s, a, tgt, done, env_data, start, reward):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(reward)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())
    data['infos/start'].append(start)

def add_next_obs(data):
    data['next_observations'] = data['observations'][1:] + data['observations'][-1:]

def npify(data):
    for k in data:
        if k == 'terminals' or k == 'timeouts':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--env_name', type=str, default='maze2d-large-dense-v1', help='Maze type')
    parser.add_argument('--starts_and_targets_path', type=str, default='st-maze2d-large-dense-v1-2.hdf5',
                        help='Path to starts and targets file')
    parser.add_argument('--controller_type', type=str, default='waypoint',
                        choices=['waypoint', 'waypoint_noisy', 'random'])
    parser.add_argument('--reward_type', type=str, default='dense', choices=['dense', 'sparse'])
    args = parser.parse_args()

    with h5py.File(args.starts_and_targets_path, "r") as f:
        starts_and_targets = f['starts_and_targets'][:]

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    # max_episode_steps = env._max_episode_steps

    env = maze_model.MazeEnv(maze_spec=maze, reward_type=args.reward_type)

    if args.controller_type == 'waypoint':
        controller = WaypointControllerWrapper(maze_str=maze, env=env)
    elif args.controller_type == 'waypoint_noisy':
        controller = NoisyWaypointController(maze_str=maze, env=env)
    elif args.controller_type == 'random':
        controller = RandomController(maze_str=maze, env=env)
    else:
        raise ValueError("controller not supported")

    data = reset_data()

    for i in range(starts_and_targets.shape[0]):

        start, target = starts_and_targets[i]
        env.set_loc(start)
        env.set_target(target)

        s = env._get_obs()
        assert np.allclose(env.sim.data.qpos, start) and np.allclose(start, s[0:2])
        assert np.allclose(env._target, target)

        act = env.action_space.sample()
        done = False
        ts = 0

        while True:
            position = s[0:2]
            velocity = s[2:4]
            act, done = controller.get_action(position, velocity, env._target)
            ns, reward, _, _ = env.step(act)
            ts += 1

            append_data(data, s, act, env._target, done, env.sim.data, start, reward)

            if args.render:
                env.render()

            if done:
                break

            else:
                s = ns

    fname = f'{args.env_name}-{args.controller_type}-{starts_and_targets.shape[0]}.hdf5'
    dataset = h5py.File(fname, 'w')
    add_next_obs(data)

    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    main()
