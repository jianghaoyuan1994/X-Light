import random

# import gfootball.env as football_env
from onpolicy.envs.sumo_files_marl.env.sim_env import TSCSimulator
# from onpolicy.envs.sumo_files_marl.config import config

from gym import spaces
import numpy as np

import torch
import copy
import os, sys
import ast

# output_path


class SUMOEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    def __init__(self, args, rank, config):
        
        self.args = args
        id = args.seed + np.random.randint(0, 2023) + rank
        self.set_seed(id)
        self.config = config
        # make env
        env_config = config['environment']
        # sumo_envs_num = len(env_config['sumocfg_files'])
        # sumo_cfg = env_config['sumocfg_files'][id % sumo_envs_num]
        if args.cotrain:
            sumo_cfg = args.sumocfg_files_list[rank]
        else:
            sumo_cfg = args.sumocfg_files
        sumo_cfg = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + sumo_cfg 
        self.length = config['mdp_length']   
        
        ###  /home/xingdp/jqruan/data/TSC/on-policy-main-sumo/onpolicy/scripts/train/onpolicy/envs/sumo_files/scenarios/large_grid2/exp_0.sumocfg
        
        env_config = copy.deepcopy(env_config)
        env_config['sumocfg_file'] = sumo_cfg
        port = args.port_start + id
        print('----port--', port, '----sumo_cfg--', sumo_cfg)
        
        # print('------------------------', port )
        if self.args.not_update:
            output_path = config.get("environment").get("eval_output_path")
            output_path = output_path + '/trial_' + str(rank) + '/'
        else:
            output_path = config.get("environment").get("output_path")
            output_path = output_path + env_config['sumocfg_files'][0].split('/')[-2] + '/trial_' + str(rank) + '/'
        if env_config['is_record'] and not os.path.exists(output_path):
            os.makedirs(output_path)
        self.env = TSCSimulator(env_config, port, output_path=output_path)
        
        self.unava_phase_index = []
        for i in self.env.all_tls:
            self.unava_phase_index.append(self.env._crosses[i].unava_index)
   
        self.num_agents = len(self.unava_phase_index)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
                
        for idx in range(self.num_agents):
            self.action_space.append(spaces.Discrete(n=env_config['num_actions']))
            self.share_observation_space.append(spaces.Box(-float('inf'), float('inf'), [env_config['obs_shape']*self.num_agents], dtype=np.float32))
            self.observation_space.append(spaces.Box(-float('inf'), float('inf'), [env_config['obs_shape']], dtype=np.float32))
        


    def get_unava_phase_index(self):
        return np.array(self.unava_phase_index)
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def get_reward(self, reward, all_tls):
        ans = []
        for i in all_tls:
            ans.append(sum(reward[i].values()))
        return np.array(ans)


    # def batch(self, env_output, use_keys, all_tl):
    #     """Transform agent-wise env_output to batch format."""
    #     if all_tl == ['gym_test']:
    #         return torch.tensor([env_output])
    #     obs_batch = {}
    #     for i in use_keys+['mask', 'neighbor_index', 'neighbor_dis']:
    #         obs_batch[i] = []
    #     for agent_id in all_tl:
    #         out = env_output[agent_id]
    #         tmp_dict = {k: np.zeros(8) for k in use_keys}
    #         state, mask, neight_msg = out
    #         for i in range(len(state)):
    #             for s in use_keys:
    #                 tmp_dict[s][i] = state[i].get(s, 0)
    #         for k in use_keys:
    #             obs_batch[k].append(tmp_dict[k])
    #         obs_batch['mask'].append(mask)
    #         obs_batch['neighbor_index'].append(neight_msg[0][0])
    #         obs_batch['neighbor_dis'].append(neight_msg[0][1])

    #     for key, val in obs_batch.items():
    #         if key not in ['current_phase', 'mask', 'neighbor_index']:
    #             obs_batch[key] = np.array(val)
    #         else:
    #             obs_batch[key] = np.array(val)
                
    #     self.obs_keys = list(obs_batch.keys())
    #     obs_values = np.hstack(list(obs_batch.values())) ### 25, 64
    #     return obs_values
    
    
    
    # def batch(self, env_output, use_keys, all_tl):
    #     """Transform agent-wise env_output to batch format."""
    #     if all_tl == ['gym_test']:
    #         return torch.tensor([env_output])
    #     obs_batch = {}
    #     for i in use_keys+['mask']:
    #         obs_batch[i] = []
    #     for agent_id in all_tl:
    #         out = env_output[agent_id]
    #         tmp_dict = {k: np.zeros(8) for k in use_keys}
    #         state, mask = out
    #         for i in range(len(state)):
    #             for s in use_keys:
    #                 tmp_dict[s][i] = state[i].get(s, 0)
    #         for k in use_keys:
    #             obs_batch[k].append(tmp_dict[k])
    #         obs_batch['mask'].append(mask)

    #     # for key, val in obs_batch.items():
    #     #     if key not in ['current_phase', 'mask']:
    #     #         obs_batch[key] = torch.FloatTensor(np.array(val))
    #     #     else:
    #     #         obs_batch[key] = torch.LongTensor(np.array(val))
        
    #     for key, val in obs_batch.items():
    #         # if key != 'pressure':
    #         if key not in ['current_phase', 'mask']:
    #             obs_batch[key] = np.array(val)
    #         else:
    #             obs_batch[key] = np.array(val)
                
    #     ###### 把 pressure 放在字典最后面
    #     obs_batch['pressure'] = obs_batch.pop('pressure')
    #     self.obs_keys = list(obs_batch.keys())
    #     obs_values = np.hstack(list(obs_batch.values())) ### 25, 56
    #     #### max-pressure
        
    #     #### A: wt-et B: el-et C: wl-wt D: el-wl E: nt-st F: sl-st G: nt-nl H: nl-sl
    #     ####    3  7     6   7    2  3      2  6    1   5    4   5    0   1     0  4
    #     # obs_batch['pressure'] 
    #     return obs_values
    
    def batch(self, env_output, use_keys, all_tl):
        """Transform agent-wise env_output to batch format."""
        if all_tl == ['gym_test']:
            return torch.tensor([env_output])
        obs_batch = {}
        for i in use_keys+['mask', 'neighbor_index', 'neighbor_dis']:
            obs_batch[i] = []
        for agent_id in all_tl:
            out = env_output[agent_id]
            tmp_dict = {k: np.zeros(8) for k in use_keys}
            state, mask, neight_msg = out
            for i in range(len(state)):
                for s in use_keys:
                    tmp_dict[s][i] = state[i].get(s, 0)
            for k in use_keys:
                obs_batch[k].append(tmp_dict[k])
            obs_batch['mask'].append(mask)
            obs_batch['neighbor_index'].append(neight_msg[0][0])
            obs_batch['neighbor_dis'].append(neight_msg[0][1])

        for key, val in obs_batch.items():
            if key not in ['current_phase', 'mask', 'neighbor_index']:
                obs_batch[key] = torch.FloatTensor(np.array(val))
            else:
                obs_batch[key] = torch.LongTensor(np.array(val))

        return obs_batch    
    
    def state_(self, state, pre_reward, pre_action, pre_done):
        neighbor_index = state['neighbor_index']
        neighbor_index = torch.cat([torch.arange(neighbor_index.shape[0]).unsqueeze(1), neighbor_index], dim=1)
        state_all = []
        for key in ['current_phase', 'car_num', 'queue_length', 'occupancy', 'flow', 'stop_car_num', 'mask', 'neighbor_index', 'neighbor_dis']:
            if key not in ['neighbor_index', 'neighbor_dis']:
                state_all.append(state[key].float())
        state_all.append(pre_reward)
        state_all.append(pre_action.float())
        state_all = torch.cat(state_all, dim=1)    # current_phase 0:8, car_num 8:16, queue_length 16:24, occupancy 24:32, flow 32:40, stop_car_num 40:48, mask 48:56, pre_reward 56:57, pre_action 58: 66
        pad_neightbor = torch.zeros((1, state_all.shape[1]))
        paded_state_all = torch.cat([state_all, pad_neightbor], dim=0)
        #  add binary indicator to classify neighbor is -1 or not
        state_self_nei = paded_state_all[neighbor_index].reshape(1, state_all.shape[0], -1)
        indicator = torch.zeros((neighbor_index.shape[0], 5))
        indicator[neighbor_index == -1] = 1
        state_self_nei = torch.cat(
            [state_self_nei, pre_done.unsqueeze(0), indicator.unsqueeze(0)], dim=-1)         # pre_done 325: 326, indicator 326:
        return state_self_nei       
    
    def reset(self):
        obs = self.env.reset()
        self.pre_reward = torch.FloatTensor(np.zeros((len(self.env.all_tls), 1)))
        self.pre_done = torch.FloatTensor(np.zeros((len(self.env.all_tls), 1)))
        self.pre_action = torch.FloatTensor(np.zeros((len(self.env.all_tls), 8)))
        self.pre_mdps_state = torch.FloatTensor(np.zeros((self.length, len(self.env.all_tls), 332)))
        obs_values = self.batch(obs, self.config['environment']['state_key'], self.env.all_tls)
        obs_values = self.state_(obs_values, self.pre_reward, self.pre_action, self.pre_done)
        obs_values = torch.cat([obs_values, torch.ones((1, len(self.env.all_tls), 1))], dim=-1)
        obs_values = self._obs_wrapper(obs_values)
        self.pre_mdps_state = self.pre_mdps_state[-self.length+1:, :, :]
        obs_values = torch.cat([self.pre_mdps_state, obs_values], dim=0)
        self.pre_mdps_state = obs_values
        return obs_values

    def step(self, action):
        tl_action_select = {}
        for tl_index in range(len(self.env.all_tls)):
            tl_action_select[self.env.all_tls[tl_index]] = \
                (self.env._crosses[self.env.all_tls[tl_index]].green_phases)[action[tl_index]]
        obs, reward, done, info = self.env.step(tl_action_select)
        reward = self.get_reward(reward, self.env.all_tls)
        reward = reward.reshape(self.num_agents, 1)
        # if self.share_reward:
        #     global_reward = np.sum(reward)
        #     reward = [[global_reward]] * self.num_agents
        
        # info['individual_reward'] = reward

        done = np.array([done] * self.num_agents)
        self.pre_done = torch.FloatTensor([done]).T
        self.pre_reward = torch.FloatTensor(reward)
        self.pre_action = torch.nn.functional.one_hot(torch.tensor(action), 8)        
        # info = self._info_wrapper(info)
        obs_values = self.batch(obs, self.config['environment']['state_key'], self.env.all_tls)
        obs_values = self.state_(obs_values, self.pre_reward, self.pre_action, self.pre_done)
        obs_values = torch.cat([obs_values, torch.ones((1, len(self.env.all_tls), 1))], dim=-1)
        obs_values = self._obs_wrapper(obs_values)
        self.pre_mdps_state = self.pre_mdps_state[-self.length+1:, :, :]
        obs_values = torch.cat([self.pre_mdps_state, obs_values], dim=0)
        self.pre_mdps_state = obs_values
        return obs_values, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.terminate()

    def _obs_wrapper(self, obs):
        if self.num_agents == 1:
            return obs[np.newaxis, :]
        else:
            return obs

    # def _info_wrapper(self, info):
    #     state = self.env.unwrapped.observation()
    #     info.update(state[0])
    #     info["max_steps"] = self.max_steps
    #     info["active"] = np.array([state[i]["active"] for i in range(self.num_agents)])
    #     info["designated"] = np.array([state[i]["designated"] for i in range(self.num_agents)])
    #     info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.num_agents)])
    #     return info
