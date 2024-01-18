import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_

import numpy as np

import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import init, check
device = 'cpu'


def np_to_torch(array):
    return torch.from_numpy(array).float().to(device)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # self.register_buffer('pe', pe)

    def forward(self, x):
        self.pe = check(self.pe).to(**self.tpdv)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayerNoLN(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayerNoLN, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayerNoLN, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderPolicy(nn.Module):
    """Transformer whose outputs are fed into a Normal distribution.
    A policy that contains a Transformer to make prediction based on a gaussian
    distribution.
    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): Minimum value for std.
        max_std (float): Maximum value for std.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.
    """

    def __init__(self, inner_ps_encoding, outer_ps_encoding,
                 obs_dim=331,
                 d_model=114,
                 dropout=0.0,
                 nhead=6,
                 num_encoder_layers=3,
                 dim_feedforward=256,
                 activation='relu',
                 obs_horizon=20,
                 policy_head_input="latest_memory",
                 tfixup=True,
                 recurrent_policy=False,
                 normalize_wm=False,
                 device='cpu',
                 pred=True):
        super(TransformerEncoderPolicy, self).__init__()
        self._obs_dim = obs_dim
        self._action_dim = 8
        self._obs_horizon = obs_horizon
        self._d_model = d_model
        self._policy_head_input = policy_head_input
        self._recurrent_policy = recurrent_policy
        self._normalize_wm = normalize_wm
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.tpdv2 = dict(dtype=torch.bool, device=device)

        self.fc_car_num = fc_block(1, 2, activation=nn.Sigmoid())
        self.fc_queue_length = fc_block(1, 2, activation=nn.Sigmoid())
        self.fc_occupancy = fc_block(1, 2, activation=nn.Sigmoid())
        self.fc_flow = fc_block(1, 2, activation=nn.Sigmoid())
        self.fc_stop_car_num = fc_block(1, 2, activation=nn.Sigmoid())
        self.fc_pre_reward = fc_block(1, 114)
        self.fc_pre_act = fc_block(8, 114)
        self.current_phase_act = nn.Sigmoid()
        self.current_phase_embedding = nn.Embedding(2, 2)
        self.mask_act = nn.Sigmoid()
        self.mask_embedding = nn.Embedding(2, 2)
        self.indicator_act = nn.Sigmoid()
        self.indicator_embedding = nn.Embedding(2, 2)
        self.pred = pred
        if self.pred:
            self.pred_fc1 = fc_block(159, 128, activation=nn.ReLU())
            self.pred_fc2 = nn.Linear(128, 114)

        self._obs_embedding = nn.Linear(
            in_features=114,
            out_features=d_model,
            bias=False
        ).to(device)

        self._inner_positional_encoding = inner_ps_encoding.to(
            next(self._obs_embedding.parameters()).device)
        self._wm_positional_encoding = outer_ps_encoding.to(
            next(self._obs_embedding.parameters()).device)

        outer_encoder_layers = TransformerEncoderLayerNoLN(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        inner_encoder_layers = TransformerEncoderLayerNoLN(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self._inner_transformer_module = nn.TransformerEncoder(inner_encoder_layers, num_encoder_layers)
        self._outer_transformer_module = nn.TransformerEncoder(outer_encoder_layers, num_encoder_layers)

        self._inner_cls_token = nn.Parameter(torch.randn(1,1,114))

        for p in self._outer_transformer_module.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        if tfixup:
            for p in self._obs_embedding.parameters():
                if p.dim() > 1:
                    torch.nn.init.normal_(p, 0, d_model ** (- 1. / 2.))

            temp_state_dic = {}
            for name, param in self._obs_embedding.named_parameters():
                if 'weight' in name:
                    temp_state_dic[name] = ((9 * num_encoder_layers) ** (- 1. / 4.)) * param

            for name in self._obs_embedding.state_dict():
                if name not in temp_state_dic:
                    temp_state_dic[name] = self._obs_embedding.state_dict()[name]
            self._obs_embedding.load_state_dict(temp_state_dic)

            temp_state_dic = {}
            for name, param in self._outer_transformer_module.named_parameters():
                if any(s in name for s in
                       ["linear1.weight", "linear2.weight", "self_attn.out_proj.weight"]):
                    temp_state_dic[name] = (0.67 * (num_encoder_layers) ** (- 1. / 4.)) * param
                elif "self_attn.in_proj_weight" in name:
                    temp_state_dic[name] = (0.67 * (num_encoder_layers) ** (- 1. / 4.)) * (
                                param * (2 ** 0.5))

            for name in self._outer_transformer_module.state_dict():
                if name not in temp_state_dic:
                    temp_state_dic[name] = self._outer_transformer_module.state_dict()[name]
            self._outer_transformer_module.load_state_dict(temp_state_dic)

        if self._policy_head_input == "latest_memory":
            self._policy_head_input_dim = d_model
        elif self._policy_head_input == "mixed_memory":  # working memory + episodic memory
            self._policy_head_input_dim = 2 * d_model
        elif self._policy_head_input == "full_memory":
            self._policy_head_input_dim = d_model * self._obs_horizon

        # self._policy_head = ActorModel(440, self._action_dim)
        # self._value_head = CriticModel(384)

        if self._recurrent_policy:
            self._memory_embedding = nn.Linear(
                in_features=d_model * self._obs_horizon,
                out_features=self._obs_dim,
                bias=False
            )

        self.src_mask = None

        # if self._normalize_wm:
        #     self.wm_rms = RunningMeanStd(shape=self._obs_dim)

        self._prev_observations = None
        self._last_hidden_state = None
        self._prev_actions = None
        self._episodic_memory_counter = None
        self._new_episode = None
        self._step = None

    def get_mask(self):
        if self.src_mask is not None:
            return self.src_mask
        sz = self._obs_horizon
        ones = torch.ones(sz, sz).to(device)
        mask = (torch.triu(ones) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.src_mask = mask
        return mask

    def forward(self, observations, unava_phase_index,  actions=None, eval=False):
        """Compute the action distributions from the observations.
        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device. Shape (S_len, B, input_step)
        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors
            torch.Tensor: Hidden States
        """
        # current_phase 0:8, car_num 8:16, queue_length 16:24, occupancy 24:32,
        # flow 32:40, stop_car_num 40:48, mask 48:56, pre_reward 56:57, pre_action 57: 65
        # pre_done 325: 326, indicator 326:


        policy_head_input, transformer_output = self.compute_memories(observations)
        # v, p = self._value_head(policy_head_input)
        # logits = self._policy_head(policy_head_input, p)
        # if unava_phase_index:
        #     for i in range(policy_head_input.shape[0]):
        #         logits[i, unava_phase_index[i]] = -1e8
        # dist = torch.distributions.Categorical(logits=logits)
        # if actions is None:
        #     if not eval:
        #         actions = dist.sample()
        #     else:
        #         actions = torch.argmax(logits, axis=1)
        # log_prob = dist.log_prob(actions).unsqueeze(-1)
        # entropy = dist.entropy().unsqueeze(-1)
        # return {'a': actions,
        #         'log_pi_a': log_prob,
        #         'ent': entropy,
        #         'v': v,
        #         'predict_o': p}
        return policy_head_input

    def compute_memories(self, observations):
        length, bs = observations.shape[:2]
        all_key_state_1 = []
        all_key_state_1.append(self.fc_car_num(observations[:, :, 8:16].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_1.append(self.fc_queue_length(observations[:, :, 16:24].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_1.append(self.fc_occupancy(observations[:, :, 24:32].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_1.append(self.fc_flow(observations[:, :, 32:40].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_1.append(self.fc_stop_car_num(observations[:, :, 40:48].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_1.append(
            self.current_phase_act(self.current_phase_embedding(observations[:, :, 0:8].long()).reshape(length, bs, -1)))
        all_key_state_1.append(
            self.mask_act(self.mask_embedding(observations[:, :, 48:56].long())).reshape(length, bs, -1))
        all_key_state_1.append(
            self.indicator_act(self.indicator_embedding(observations[:, :, 326].long())).reshape(length, bs, -1))
        obs_1 = torch.cat(all_key_state_1, dim=-1).reshape(length*bs, 1, -1)

        all_key_state_2 = []
        all_key_state_2.append(self.fc_car_num(observations[:, :, 65+8:65+16].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_2.append(self.fc_queue_length(observations[:, :, 65+16:65+24].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_2.append(self.fc_occupancy(observations[:, :, 65+24:65+32].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_2.append(self.fc_flow(observations[:, :, 65+32:65+40].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_2.append(self.fc_stop_car_num(observations[:, :, 65+40:65+48].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_2.append(
            self.current_phase_act(self.current_phase_embedding(observations[:, :, 65+0:65+8].long()).reshape(length, bs, -1)))
        all_key_state_2.append(
            self.mask_act(self.mask_embedding(observations[:, :, 65+48:65+56].long())).reshape(length, bs, -1))
        all_key_state_2.append(
            self.indicator_act(self.indicator_embedding(observations[:, :, 327].long())).reshape(length, bs, -1))
        obs_2 = torch.cat(all_key_state_2, dim=-1).reshape(length*bs, 1, -1)

        all_key_state_3 = []
        all_key_state_3.append(self.fc_car_num(observations[:, :, 130+8:130+16].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_3.append(self.fc_queue_length(observations[:, :, 130+16:130+24].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_3.append(self.fc_occupancy(observations[:, :, 130+24:130+32].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_3.append(self.fc_flow(observations[:, :, 130+32:130+40].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_3.append(self.fc_stop_car_num(observations[:, :, 130+40:130+48].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_3.append(
            self.current_phase_act(self.current_phase_embedding(observations[:, :, 130+0:130+8].long()).reshape(length, bs, -1)))
        all_key_state_3.append(
            self.mask_act(self.mask_embedding(observations[:, :, 130+48:130+56].long())).reshape(length, bs, -1))
        all_key_state_3.append(
            self.indicator_act(self.indicator_embedding(observations[:, :, 328].long())).reshape(length, bs, -1))
        obs_3 = torch.cat(all_key_state_3, dim=-1).reshape(length*bs, 1, -1)

        all_key_state_4 = []
        all_key_state_4.append(self.fc_car_num(observations[:, :, 195+8:195+16].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_4.append(self.fc_queue_length(observations[:, :, 195+16:195+24].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_4.append(self.fc_occupancy(observations[:, :, 195+24:195+32].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_4.append(self.fc_flow(observations[:, :, 195+32:195+40].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_4.append(self.fc_stop_car_num(observations[:, :, 195+40:195+48].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_4.append(
            self.current_phase_act(self.current_phase_embedding(observations[:, :, 195+0:195+8].long()).reshape(length, bs, -1)))
        all_key_state_4.append(
            self.mask_act(self.mask_embedding(observations[:, :, 195+48:195+56].long())).reshape(length, bs, -1))
        all_key_state_4.append(
            self.indicator_act(self.indicator_embedding(observations[:, :, 329].long())).reshape(length, bs, -1))
        obs_4 = torch.cat(all_key_state_4, dim=-1).reshape(length*bs, 1, -1)

        all_key_state_5 = []
        all_key_state_5.append(self.fc_car_num(observations[:, :, 260+8:260+16].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_5.append(self.fc_queue_length(observations[:, :, 260+16:260+24].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_5.append(self.fc_occupancy(observations[:, :, 260+24:260+32].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_5.append(self.fc_flow(observations[:, :, 260+32:260+40].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_5.append(self.fc_stop_car_num(observations[:, :, 260+40:260+48].reshape(-1, 1)).reshape(length, bs, -1))
        all_key_state_5.append(
            self.current_phase_act(self.current_phase_embedding(observations[:, :, 260+0:260+8].long()).reshape(length, bs, -1)))
        all_key_state_5.append(
            self.mask_act(self.mask_embedding(observations[:, :, 260+48:260+56].long())).reshape(length, bs, -1))
        all_key_state_5.append(
            self.indicator_act(self.indicator_embedding(observations[:, :, 330].long())).reshape(length, bs, -1))
        obs_5 = torch.cat(all_key_state_5, dim=-1).reshape(length*bs, 1, -1)

        pre_reward1 = self.fc_pre_reward(observations[:, :, 56:57].reshape(-1, 1)).reshape(length*bs, 1, -1)
        pre_reward2 = self.fc_pre_reward(observations[:, :, 65+56:65+57].reshape(-1, 1)).reshape(length*bs, 1, -1)
        pre_reward3 = self.fc_pre_reward(observations[:, :, 130+56:130+57].reshape(-1, 1)).reshape(length*bs, 1, -1)
        pre_reward4 = self.fc_pre_reward(observations[:, :, 195+56:195+57].reshape(-1, 1)).reshape(length*bs, 1, -1)
        pre_reward5 = self.fc_pre_reward(observations[:, :, 260+56:260+57].reshape(-1, 1)).reshape(length*bs, 1, -1)

        pre_act1 = self.fc_pre_act(observations[:, :, 57:65].reshape(-1, 8)).reshape(length*bs, 1, -1)
        pre_act2 = self.fc_pre_act(observations[:, :, 65+57:65+65].reshape(-1, 8)).reshape(length*bs, 1, -1)
        pre_act3 = self.fc_pre_act(observations[:, :, 130+57:130+65].reshape(-1, 8)).reshape(length*bs, 1, -1)
        pre_act4 = self.fc_pre_act(observations[:, :, 195+57:195+65].reshape(-1, 8)).reshape(length*bs, 1, -1)
        pre_act5 = self.fc_pre_act(observations[:, :, 260+57:260+65].reshape(-1, 8)).reshape(length*bs, 1, -1)
        # all_key_state.append(self.fc_pre_reward(observations[:, :, 56:57].reshape(-1, 1)).reshape(length, bs, -1))
        # all_key_state.append(observations[:, :, 57:65])
        inner_obs = torch.cat([self._inner_cls_token.expand(obs_1.shape), obs_1, pre_reward1, pre_act1, obs_2, pre_reward2, pre_act2,
                               obs_3, pre_reward3, pre_act3, obs_4, pre_reward4, pre_act4,
                               obs_5, pre_reward5, pre_act5], dim=1).permute(1, 0, 2)
        in_pos = self._inner_positional_encoding(inner_obs)
        inner_transformer_output = self._inner_transformer_module(in_pos)[0]
        # Get original shapes and reshape tensors to have a single batch dimension
        # obs_shape = list(observations.shape)
        # batch_shape = obs_shape[:-2]
        # observations = torch.reshape(observations, (-1, obs_shape[-2], obs_shape[-1]))

        # Computing working memory as a representation from tuple (obs, act, rew)
        #if self.pred:
        #    pred_ele = inner_transformer_output.reshape(length, bs, -1)
        #    pred_input = torch.cat([pred_ele[:-1],
        #                            observations[1:, :, 56:57], observations[1:, :, 65+56:65+57],
        #                            observations[1:, :, 130 + 56:130 + 57], observations[1:, :, 195+56:195+57],
        #                            observations[1:, :, 260 + 56:260 + 57],
        #                            observations[1:, :, 57:65], observations[1:, :, 65+57:65+65],
        #                            observations[1:, :, 130 + 57:130 + 65], observations[1:, :, 195+57:195+65],
        #                            observations[1:, :, 260 + 57:260 + 65]
        #                            ], dim=-1)
        #    pred = self.pred_fc2(self.pred_fc1(pred_input))
        #else:
        #    pred = [0,0]
        #    pred_ele = [0,0]

        working_memo = self._obs_embedding(inner_transformer_output)  # (B, S_len, output_step)
        # working_memo = working_memo * math.sqrt(self._d_model)

        # get memory index
        # curr_em_index = self._compute_memory_index(observations).\
        #     unsqueeze(-1).repeat(1, working_memo.shape[-1]).unsqueeze(1)

        # Get current working memory as the most recent in the tensor
        # curr_working_memo = torch.gather(working_memo, dim=1, index=curr_em_index)
        working_memo = working_memo.reshape(length, bs, -1)
        # working_memo = working_memo.permute(1, 0, 2)  # Transformer module inputs (S_len, B, output_step)
        wm_pos = self._wm_positional_encoding(working_memo)
        pad_mask = 1 - observations[:,:,-1].permute(1,0)
        transformer_output = self._outer_transformer_module(wm_pos, src_key_padding_mask=check(pad_mask).to(**self.tpdv2))  # (T, B, target_output)
        transformer_output = transformer_output.permute(1, 0, 2)  # going back to batch first


        if self.pred:
            pred_ele = transformer_output.permute(1, 0, 2).reshape(length, bs, -1)
            # pred_ele = transformer_output.reshape(length, bs, -1)
            pred_input = torch.cat([pred_ele[:-1],
                                    observations[1:, :, 56:57],
                                    observations[1:, :, 65 + 56:65 + 57],
                                    observations[1:, :, 130 + 56:130 + 57],
                                    observations[1:, :, 195 + 56:195 + 57],
                                    observations[1:, :, 260 + 56:260 + 57],
                                    observations[1:, :, 57:65],
                                    observations[1:, :, 65 + 57:65 + 65],
                                    observations[1:, :, 130 + 57:130 + 65],
                                    observations[1:, :, 195 + 57:195 + 65],
                                    observations[1:, :, 260 + 57:260 + 65]
                                    ], dim=-1)
            pred = self.pred_fc2(self.pred_fc1(pred_input))
        else:
            pred = [0, 0]
            pred_ele = [0, 0]

        # Compute policy head input
        if self._policy_head_input == "full_memory":
            return torch.reshape(transformer_output, bs + [
                self._policy_head_input_dim]), transformer_output.detach().cpu().numpy()

        # curr_em = torch.gather(transformer_output, dim=1, index=curr_em_index)
        #curr_em = transformer_output[:, -1, :]
        curr_em = transformer_output[:, -1, :] + obs_1.reshape(length, bs, -1).permute(1, 0, 2)[:,-1,:]
        final_shape_obs = [bs] + [self._obs_embedding.out_features]
        curr_em = torch.reshape(curr_em, final_shape_obs)  # get just the last hidden state as input for policy head
        # curr_working_memo = torch.reshape(curr_working_memo, final_shape_obs)

        memories = None
        if self._policy_head_input == "latest_memory":
            memories = curr_em
        # elif self._policy_head_input == "mixed_memory":
        #     memories = torch.cat((curr_working_memo, curr_em), axis=-1)

        # if self._recurrent_policy:
        #     transformer_output = self._memory_embedding(
        #         torch.reshape(transformer_output, batch_shape + [self._d_model * self._obs_horizon]))

        return memories, transformer_output.detach().cpu().numpy(), pred, pred_ele[1:], observations[:-1, :, -1]

def sequential_pack(layers):
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in layers:
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq


def fc_block(
        in_channels,
        out_channels,
        activation=None,
        use_dropout=False,
        norm_type=None,
        dropout_probability=0.5,
        bias=True
):
    block = [nn.Linear(in_channels, out_channels, bias=bias)]
    xavier_normal_(block[-1].weight)
    if norm_type is not None and norm_type != 'none':
        if norm_type == 'LN':
            block.append(nn.LayerNorm(out_channels))
        else:
            raise NotImplementedError
    if isinstance(activation, torch.nn.Module):
        block.append(activation)
    elif activation is None:
        pass
    else:
        raise NotImplementedError
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)
    
class ActorModel(nn.Module):
    def __init__(self, hidden_size, action_size):
        super(ActorModel, self).__init__()
        self.name = 'actor'
        self.model = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), action_size),
        )

    def forward(self, hidden_states):
        outputs = self.model(hidden_states)
        return outputs


class CriticModel(nn.Module):
    def __init__(self, hidden_size):
        super(CriticModel, self).__init__()
        self.name = 'critic'
        self.c_fc1 = fc_block(hidden_size, int(hidden_size / 2), activation=nn.ReLU())
        self.c_fc2 = nn.Linear(hidden_size, 1)
        self.p_fc1 = fc_block(hidden_size, int(hidden_size / 2), activation=nn.ReLU())
        self.p_fc2 = nn.Linear(int(hidden_size / 2), 56)

    def forward(self, hidden_states):
        c1 = self.c_fc1(hidden_states)
        p1 = self.p_fc1(hidden_states)
        c2 = self.c_fc2(torch.cat([c1, p1], dim=-1))
        p2 = self.p_fc2(p1)
        return c2, p2
