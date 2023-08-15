#!/usr/bin/env python3
# encoding: utf-8
name = 'TIT_mdp_pad_mask_pred3_cls_token_addind_3'
config = {
    'name': name,
    "episode": {
        "rollout_length": 240,
    },
    "ppo": {
        "value_loss_coef": 1.0,
        'predict_o_loss': 0.1
    },
    "environment": {
        "num_actions": 8,
        "obs_shape": 332,
        "action_type": "select_phase",
        "is_dis": True,
        "is_libsumo": True,
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        "is_record": False,
        "is_neighbor_reward": False,
        'output_path': 'onpolicy/envs/sumo_files_marl/sumo_logs/{}/'.format(name),
        'eval_output_path': 'onpolicy/envs/sumo_files_marl/sumo_logs/eval_{}/'.format(name),
        "name": name,
        'port_start': 16900, # grid4x4
        "sumocfg_files": [
            "sumo_files_marl/scenarios/large_grid2/exp_0.sumocfg",
            'sumo_files_marl/scenarios/resco_envs/grid4x4/grid4x4.sumocfg'
            "sumo_files_marl/scenarios/sumo_fenglin_base_road/base.sumocfg"
            "sumo_files_marl/scenarios/nanshan/osm.sumocfg"
            'sumo_files_marl/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg'
            'sumo_files_marl/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg'
            'sumo_files_marl/scenarios/resco_envs/cologne8/cologne8.sumocfg'
        ],
        "state_key": ['current_phase', 'car_num', 'queue_length', "occupancy", 'flow', 'stop_car_num'],
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score']
    },
    "model_save": {
        "frequency": 200,
        "path": "onpolicy/envs/sumo_files_new/tsc/{}".format(name)
    },
    'mdp_length': 20
}
