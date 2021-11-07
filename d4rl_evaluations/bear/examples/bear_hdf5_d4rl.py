from pointmaze import MazeEnv
from gym.envs.mujoco import HalfCheetahEnv, HopperEnv, AntEnv, Walker2dEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, VAEPolicy
from rlkit.torch.sac.bear import BEARTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import numpy as np

import h5py, argparse, os
import gym
import d4rl

def load_hdf5(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size

# def load_hdf5(dataset, replay_buffer, max_size):
#     all_obs = dataset['observations']
#     all_act = dataset['actions']
#     N = min(all_obs.shape[0], max_size)
#
#     _obs = all_obs[:N-1]
#     _actions = all_act[:N-1]
#     _next_obs = all_obs[1:]
#     _rew = np.squeeze(dataset['rewards'][:N-1])
#     _rew = np.expand_dims(np.squeeze(_rew), axis=-1)
#     _done = np.squeeze(dataset['terminals'][:N-1])
#     _done = (np.expand_dims(np.squeeze(_done), axis=-1)).astype(np.int32)
#
#     max_length = 1000
#     ctr = 0
#     ## Only for MuJoCo environments
#     ## Handle the condition when terminal is not True and trajectory ends due to a timeout
#     for idx in range(_obs.shape[0]):
#         if ctr  >= max_length - 1:
#             ctr = 0
#         else:
#             replay_buffer.add_sample_only(_obs[idx], _actions[idx], _rew[idx], _next_obs[idx], _done[idx])
#             ctr += 1
#             if _done[idx][0]:
#                 ctr = 0
#     ###
#
#     print (replay_buffer._size, replay_buffer._terminals.shape)


def experiment(variant):
    expl_env = gym.make(variant['env_name'])

    with h5py.File(variant['starts_and_targets_path'], "r") as f:
        starts_and_targets = f['starts_and_targets'][:]
    maze = expl_env.str_maze_spec
    eval_env = MazeEnv(maze_spec=maze, starts_and_targets=starts_and_targets)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M,],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M,],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M,],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M,],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M,], 
    )
    vae_policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[750, 750],
        latent_dim=action_dim * 2,
    )
    eval_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    if variant['load_buffer'] and buffer_filename is not None:
        dataset = expl_env.get_dataset(buffer_filename)
        load_hdf5(dataset, replay_buffer)#, max_size=variant['replay_buffer_size'])
    else:
        load_hdf5(eval_env.unwrapped.get_dataset(), replay_buffer)#, max_size=variant['replay_buffer_size'])
    
    trainer = BEARTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vae=vae_policy,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        batch_rl=True,
        q_learning_alg=True,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='BEAR-runs')
    parser.add_argument("--env", type=str, default='maze2d-large-dense-v1')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument('--qf_lr', default=3e-4, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)
    parser.add_argument('--mmd_sigma', default=50, type=float)
    parser.add_argument('--kernel_type', default='gaussian', type=str)
    parser.add_argument('--target_mmd_thresh', default=0.05, type=float)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--buffer_filename', type=str,
                        default='/home/kg/Projects/off/d4rl/d4rl/maze2d-large-dense-v1-waypoint-1.hdf5')
    parser.add_argument('--starts_and_targets_path', type=str,
                        default='/home/kg/Projects/off/d4rl/d4rl/st-maze2d-large-dense-v1-2.hdf5',
                        help='Path to starts and targets file')
    args = parser.parse_args()

    variant = dict(
        algorithm="BEAR",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=args.buffer_filename, #halfcheetah_101000.pkl',
        load_buffer=True,
        env_name=args.env,
        starts_and_targets_path=args.starts_and_targets_path,
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            num_actions_sample=args.num_samples,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=1,

            # BEAR specific params
            mode='auto',
            kernel_choice=args.kernel_type,
            policy_update_style='0',
            mmd_sigma=args.mmd_sigma,
            target_mmd_thresh=args.target_mmd_thresh,

        ),
    )
    rand = np.random.randint(0, 100000)
    setup_logger(os.path.join('BEAR_launch', str(rand)), variant=variant, base_log_dir='/home/kg/Projects/off/d4rl_evaluations/bear/data')
    ptu.set_gpu_mode(False)  # optionally set the GPU (default=False)
    experiment(variant)
