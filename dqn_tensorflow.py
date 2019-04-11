import gym
import numpy as np
from coinrun import setup_utils, make, wrappers
from baselines import deepq
import tensorflow as tf
import matplotlib.pyplot as plt





CHECKPOINT_NAME = "coinrun_agent.pkl"

class CoinRunVecEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        obss, _, _, _ = self.env.step_wait()
        return obss[0]
        
    
    def step(self, action):
        (obss, rews, dones, infos) = self.env.step(np.array([action]))
        return obss[0], rews[0], dones[0], infos


def make_model(network = 'cnn', **network_kwargs):
    from baselines.common.models import get_network_builder
    return get_network_builder(network)(**network_kwargs)



def run_deepq(model, env, checkpoint=None, total_timesteps=100000, name=""):
    statistics = dict(rewards=[], t=[])

    def callback(lcl, _glb):

        statistics['rewards'].append(lcl['episode_rewards'][0])
        statistics['t'].append(lcl['t'])
        

    
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
        load_path=checkpoint,
        reuse=tf.AUTO_REUSE
    )

    

    act.save(CHECKPOINT_NAME)

    return statistics

def plot_stats(base_stats, transfer_stats, base_mode_transfer_env_stats):
    plt.figure(1)
    plt.plot(base_stats['t'], base_stats['rewards'], color='r')
    plt.xlabel('t')
    plt.ylabel('reward')
    plt.title('Base model reward')
    plt.savefig("base_plot.jpg")
    #plt.show()
    


    plt.figure(2)

    plt.plot(transfer_stats['t'], transfer_stats['rewards'], color='r', label="retrained model")
    plt.plot(base_mode_transfer_env_stats['t'], base_mode_transfer_env_stats['rewards'], color='b', label="fresh model")
    plt.xlabel('t')
    plt.ylabel('Reward')
    plt.title('Retrained model and fresh model reward')
    plt.savefig("retrained_plot.jpg")
    #plt.show()

    

def main():
    
    setup_utils.setup_and_load(use_cmd_line_args=False)

    # Make the base enviroments that we will train the agent on first, this makes 1 gym enviroment
    # But after each epoch a different enviroment will be choosen, currently we only use 1 enviroment, 
    # because it works better with the dqn algorithm
    base_env = make('standard', num_envs = 1)
    base_env = CoinRunVecEnvWrapper(base_env)

    #base_env = wrappers.add_final_wrappers(base_env)
    # Make the enviroment that we will attempt to transfer to
    transfer_enviroment = make('standard', num_envs = 1)
    transfer_enviroment = CoinRunVecEnvWrapper(transfer_enviroment)

    t = int(5e4)
    with tf.Session():
        model = make_model()

        print("-----\ntraining base model on training enviroment\n-----")
        base_statistics = run_deepq(model if model else 'cnn', base_env, total_timesteps=t, name="base")
        
        print('mean reward: ', np.mean(np.array(base_statistics['rewards'])))
    
        print("-----\ntraining transfer model on test enviroment\n-----")
        transfer_statistics = run_deepq(model if model else 'cnn', transfer_enviroment, total_timesteps=t, name="transfer")
        print('mean reward: ', np.mean(np.array(transfer_statistics['rewards'])))
    #with tf.Session():
        #with tf.variable_scope("non-transfer-model"):
        model = make_model()
        print("-----\ntraining non-transfer model on test enviroment\n-----")
        transfer_enviroment_base_model_statistics = run_deepq(model if model else 'cnn', transfer_enviroment, total_timesteps=t, name="transfer")
        print('mean reward: ', np.mean(np.array(transfer_enviroment_base_model_statistics['rewards'])))
        plot_stats(base_statistics, transfer_statistics, transfer_enviroment_base_model_statistics)




if __name__ == '__main__':
    main()
