import wandb
import os
import re
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer, SharedReplayBufferSUMO
from onpolicy.algorithms.utils.util import check


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class SUMOBaseRunner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):
        self.begin_ep = 0
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.meta_test = config['meta_test']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = os.path.join(self.run_dir, 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            if self.all_args.cotrain:
                self.writters = [SummaryWriter(self.log_dir+'/'+self.all_args.sumocfg_files_list[i].split('/')[-2]) for i in range(len(self.all_args.sumocfg_files_list))]
            self.save_dir = os.path.join(self.run_dir, 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO_SUMO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy_SUMO as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy = Policy(config,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)


        # algorithm
        self.trainer = TrainAlgo(config, self.policy, device = self.device)
        if self.model_dir is not None:
            file_names = os.listdir(self.model_dir)
            digit_pattern = r'\d+'
            digits = set()
            for file_name in file_names:
                match = re.search(digit_pattern, file_name)
                if match is not None:
                    digits.add(int(match.group(0)))
            if len(digits) > 0:
                if self.meta_test:
                    self.restore2(self.model_dir, self.all_args.model_interval)
                else:
                    self.restore(self.all_args.model_interval)
                    if not self.all_args.not_update:
                        self.begin_ep = int(self.all_args.model_interval) + 1

               
        # # buffer
        # self.buffer = SharedReplayBuffer(self.all_args,
        #                                 self.num_agents,
        #                                 self.envs.observation_space[0],
        #                                 share_observation_space,
        #                                 self.envs.action_space[0])
        # buffer
        self.buffer = SharedReplayBufferSUMO(self.all_args,
                                        config['env_config']['mdp_length'],
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        # obs = check(np.concatenate(self.buffer.obs[-1], axis = 1)).to(**self.tpdv)
        hidden_states = check(np.concatenate(self.buffer.hidden_layer[-1])).to(**self.tpdv)
        next_values = self.trainer.policy.get_values(hidden_states)
        # next_values = next_values.detach()
        
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self, episode):
        """Save policy's actor and critic networks."""
        policy_shared_NN = self.trainer.policy.shared_NN
        torch.save(policy_shared_NN.state_dict(), str(self.save_dir) + f"/shared_NN_{episode}.pt")
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + f"/actor_{episode}.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + f"/critic_{episode}.pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + f"/vnorm_{episode}.pt")
        print(
            "Save model done: {}".format(str(self.save_dir) + f'/shared_NN_{episode}.pt'))

    def restore(self, model_interval):
        """Restore policy's networks from a saved model."""
        policy_shared_NN_state_dict = torch.load(str(self.model_dir) + f'/shared_NN_{model_interval}.pt')
        self.policy.shared_NN.load_state_dict(policy_shared_NN_state_dict)
        policy_actor_state_dict = torch.load(str(self.model_dir) + f'/actor_{model_interval}.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + f'/critic_{model_interval}.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(str(self.model_dir) + f'/vnorm_{model_interval}.pt')
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)
        print("Load model done: {}".format(str(self.model_dir) + f'/shared_NN_{model_interval}.pt'))

    def restore2(self, path, model_interval):
        policy_shared_NN_state_dict = torch.load(path + f'/shared_NN_{model_interval}.pt', map_location=torch.device('cpu'))
        self.policy.shared_NN.load_state_dict(policy_shared_NN_state_dict)
        policy_actor_state_dict = torch.load(path + f'/actor_{model_interval}.pt', map_location=torch.device('cpu'))
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(path + f'/critic_{model_interval}.pt', map_location=torch.device('cpu'))
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(path + f'/vnorm_{model_interval}.pt', map_location=torch.device('cpu'))
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)
        print("Load model done: {}".format(path + f'/shared_NN_{model_interval}.pt'))
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        # for k, v in env_infos.items():
        #     if len(v)>0:
        #         if self.use_wandb:
        #             wandb.log({k: np.mean(v)}, step=total_num_steps)
        #         else:
        #             self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
        if not self.all_args.cotrain:
            total_reward = 0            
            for k, v in env_infos.items():
                self.writter.add_scalar('env_info_reward/{}'.format(k), np.mean(v), total_num_steps)
                total_reward += np.mean(v)
                # print('Step{}, reward_{}: {}'.format(step, k, np.mean(v)))
            self.writter.add_scalar('env_info_reward/all', total_reward, total_num_steps)
        else:
            for i in range(len(env_infos)):
                env_info = env_infos[i]
                total_reward = 0            
                for k, v in env_info.items():
                    self.writters[i].add_scalar('env_info_reward/{}'.format(k), np.mean(v), total_num_steps)
                    total_reward += np.mean(v)
                    # print('Step{}, reward_{}: {}'.format(step, k, np.mean(v)))
                self.writters[i].add_scalar('env_info_reward/all'.format(k), total_reward, total_num_steps)



class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_0.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_0.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm.pt')
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
