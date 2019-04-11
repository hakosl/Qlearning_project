import gym
import numpy as np
from coinrun import setup_utils, make, wrappers
from baselines import deepq
import tensorflow as tf



def make_model():
    return None

CHECKPOINT_NAME = "coinrun_agent.pkl"

def run_deepq(model, env, checkpoint=None, total_timesteps=100000):
    statistics = dict(rewards=[], loss=[], timesteps=[])

    def callback(lcl, _glb):
        statistics['rewards'].append(lcl['episode_rewards'])

    
    act = deepq.learn(
        env,
        network=model,
        lr=1e-3,
        total_timesteps=total_timesteps,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        #load_path=checkpoint
    )

    act.save(CHECKPOINT_NAME)

    return statistics

def main():

    setup_utils.setup_and_load(use_cmd_line_args=False)

    # Make the base enviroments that we will train the agent on first, this makes 1 gym enviroment
    # But after each epoch a different enviroment will be choosen
    base_env = make('standard', num_envs = 4)
    base_env = wrappers.add_final_wrappers(base_env)
    # Make the enviroment that we will attempt to transfer to
    transfer_enviroment = make('standard', num_envs = 1)


    with tf.Session():
        model = make_model()

        print("-----\ntraining base model on training enviroment\n-----")
        base_statistics = run_deepq(model if model else 'cnn', base_env, total_timesteps=int(10e4))
        print("-----\ntraining transfer model on test enviroment\n-----")
        transfer_statistics = run_deepq(model if model else 'cnn', transfer_enviroment, checkpoint=CHEKPOINT_NAME, total_timesteps=int(10e4))
        print("-----\ntraining transfer model on test enviroment\n-----")
        transfer_enviroment_base_model_statistics = run_deepq(model if model else 'cnn', transfer_enviroment, checkpoint=CHEKPOINT_NAME, total_timesteps=int(10e4))




if __name__ == '__main__':
    main()
