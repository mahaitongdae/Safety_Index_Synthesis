#!/usr/bin/env python
import gym 
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
from safe_rl.utils.custom_env_utils import register_custom_env


def main(env_name, algo, seed, exp_name, cpu):

    register_custom_env()

    # Verify experiment

    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo', 'ppo_dual_ascent']

    algo = algo.lower()
    assert algo in algo_list, "Invalid algo"

    # Hyperparameters
    exp_name = algo + '_' + env_name
    # if robot=='Doggo':
    #     num_steps = 1e8
    #     steps_per_epoch = 60000
    # else:
    num_steps = 2e6
    steps_per_epoch = 8000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 0

    # Fork for parallelizing
    mpi_fork(cpu, bind_to_core=True)

    # Prepare Logger
    exp_name = exp_name # or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    # env_name = 'Safexp-'+robot+task+'-v0'

    algo(env_fn=lambda: gym.make(env_name),
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )

def main2(algo, seed, exp_name, cpu):
    num_steps = 2e6
    steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25
    mpi_fork(cpu)
    # Prepare Logger
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.' + algo)
    env_name = 'CrossroadEnd2end-v0'

    algo(env_fn=lambda: gym.make(env_name),
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
         ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Safexp-CustomPush1-v0')
    parser.add_argument('--algo', type=str, default='ppo_dual_ascent')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='for exp')
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.env_name, args.algo, args.seed, exp_name, args.cpu)
