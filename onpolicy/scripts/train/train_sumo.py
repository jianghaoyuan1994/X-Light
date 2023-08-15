#!/usr/bin/env python
import copy
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
# from onpolicy.envs.mpe.MPE_env import MPEEnv
# from onpolicy.envs.sumo_files.SUMO_env import SUMOEnv

# from onpolicy.envs.sumo_files_new.SUMO_env import SUMOEnv
# from onpolicy.envs.sumo_files_new.config import config as config_env

from onpolicy.envs.sumo_files_marl.SUMO_env import SUMOEnv
from onpolicy.envs.sumo_files_marl.config import config as config_env


from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

"""Train script for MPEs."""

def make_train_env(all_args, env_config):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SUMO":
                env = SUMOEnv(all_args, rank, env_config)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.cotrain:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)], cotrain=all_args.cotrain)
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SUMO":
                env = SUMOEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=0, help="the number of the agents.")
    parser.add_argument('--scenario_name', type=str, default='sumo_test', help="Which scenario to run on")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args, meta_test=False):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    #########################################
    import ast
    if all_args.cotrain:
        all_args.sumocfg_files_list = ast.literal_eval(all_args.sumocfg_files)
    all_args.episode_length = config_env['episode']['rollout_length']
    all_args.num_actions = config_env['environment']['num_actions']
    

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
    #     str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))
    setproctitle.setproctitle("bq-test")
    
    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                0] + "/results_sumo") / all_args.env_name / all_args.experiment_name / all_args.algorithm_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                        project=all_args.env_name,
                        entity=all_args.user_name,
                        notes=socket.gethostname(),
                        name=str(all_args.algorithm_name) + "_" +
                        str(all_args.experiment_name) +
                        "_seed" + str(all_args.seed),
                        group=all_args.scenario_name,
                        dir=str(run_dir),
                        job_type="training",
                        reinit=True)
    else:
        curr_run = 'seed_' + str(all_args.seed) + '_'
        if not run_dir.exists():
            curr_run += 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run += 'run1'
            else:
                curr_run += 'run%i' % (max(exst_run_nums) + 1)


        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    all_args.model_dir = os.path.join(run_dir, "models")
    if meta_test:
        run_dir = os.path.join(run_dir, eval_model_path[index])
    if not os.path.exists(run_dir):
        os.makedirs(str(run_dir))
            
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args, config_env)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    all_args.num_agents = len(envs.action_space)
    
    if all_args.cotrain:
        all_args.n_rollout_threads = 1
        all_args.n_training_threads = 1
        all_args.map_agents = envs.map_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
        'env_config': config_env,
        "meta_test": meta_test
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.sumo_runner import SUMORunner as Runner

    runner = Runner(config)
    runner.run()
    # if not all_args.use_test:
    #     runner.run()
    # else:
    #     runner.test()
    # post process
    envs.close()
    # if all_args.use_eval and eval_envs is not envs:
    #     eval_envs.close()

    # if not all_args.use_test:
    #     if all_args.use_wandb:
    #         run.finish()
    #     else:
    #         runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    #         runner.writter.close()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    for i in range(0, 1):
        args = ['--env_name', 'SUMO', '--algorithm_name', 'ippo', '--num_agents', '0', '--seed', '1',  '--experiment_name', 'grid4x4-frap',
                '--n_training_threads', '8', '--n_rollout_threads', '2', '--num_mini_batch', '1', '--num_env_steps', '750000',
                '--ppo_epoch', '5', '--gain', '0.01', '--lr', '1e-4', '--critic_lr', '1e-4', '--use_wandb', 'False', '--use_ReLU']
        index = 0
        eval_model_path = ['grid4x4', 'fenglin', 'nanshan', 'arterial4x4', 'ingolstadt21', 'cologne8','grid5x5'] 
        
        cong_lst = ['cotrain_test', 'grid4x4-test', 'fenglin-test', 'nanshan-test', 'arterial4x4-test', 'ingolstadt21-test', 'cologne8-test', 'exp_0']
        
        
        ################# 暂时不改
        threads = ['2', '32', '64', '128']
        num_mini_batch = ['8', '32', '16', '16']
        index_ = 0
        args[13] = threads[index_] # n_rollout_threads
        args[11] = threads[index_] # n_training_threads
        args[15] = num_mini_batch[index_] # num_mini_batch
        
        args[3] = 'ippo'
        # args[3] = 'mappo'
        
        args[7] = str(i)
        # args[9] = cong_lst[index] + '_thr_' + args[13]
        args[9] = config_env['name']

        ################### config ###########################
        port_lst = ['15000', '15100', '15200', '15300', '15400', '15500', '156000', '15700']
        sumocfg_files_lst = [
            'sumo_files_marl/scenarios/resco_envs/grid4x4/grid4x4.sumocfg',
            "sumo_files_marl/scenarios/sumo_fenglin_base_road/base.sumocfg",
            "sumo_files_marl/scenarios/nanshan/osm.sumocfg",
            'sumo_files_marl/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg',
            'sumo_files_marl/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg',
            'sumo_files_marl/scenarios/resco_envs/cologne8/cologne8.sumocfg',
            'sumo_files_marl/scenarios/large_grid2/exp_0.sumocfg',
            'sumo_files_marl/scenarios/large_grid2/exp_1.sumocfg'
        ] 
        state_key = config_env['environment']['state_key']
        ########################################  for training
        cotrain = True
        meta_test=False

        # cotrain = False
        # meta_test = True

        if cotrain:
           args[11] = str(len(sumocfg_files_lst))
           args[13] = str(len(sumocfg_files_lst))
           args.extend(['--state_key', state_key, '--port_start', port_lst[index], '--sumocfg_files', sumocfg_files_lst, '--cotrain', 'True'])
        else:
           args.extend(['--state_key', state_key, '--port_start', port_lst[index], '--sumocfg_files', sumocfg_files_lst[index]])
        # args.extend(['--model_interval', '108'])  # for load model
        main(args, meta_test=meta_test)
        ################################################################
        
        ######################################## For evalation
      #   import re
      #   config_env['environment']['is_record'] = True
      #   model_dir = 'onpolicy/scripts/results_sumo/SUMO/{}/ippo/seed_0_run1/models'.format(config_env['name'])
      #   args.extend(['--model_dir', model_dir])
      #   file_names = os.listdir(model_dir)
      #   digit_pattern = r'\d+'
      #   digits = set()
      #   for file_name in file_names:
      #       match = re.search(digit_pattern, file_name)
      #       if match is not None:
      #           digits.add(int(match.group(0)))
      #   result = sorted(list(digits))
      #   for index in range(len(sumocfg_files_lst)):
      #       args_ = copy.deepcopy(args)#控制评测哪张图
      #       args_.extend(['--state_key', state_key, '--port_start', port_lst[index], '--sumocfg_files', sumocfg_files_lst[index]])
      #       args_.extend(['--not_update', 'True'])  #### 如果使用评估的话就不更新模型了
      #       args_[13] = '5' # n_rollout_threads #多线程评测，要评测几次就填几
      #       args_[11] = '5' # n_training_threads
      # #
      #       for i in range(0, len(result),5):
      #           args_.extend(['--model_interval', str(result[i])])
      #           config_env['environment']['p'] = 2 * i + 1
      #           main(args_, False)
