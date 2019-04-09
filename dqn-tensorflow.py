import gym
import tensorflow as tf


env = gym.make('CartPole-v1')
observation_size = env.observation_space.shape[0]

def dense_nn(inputs, layer_sizes, scope_name):
    with tf.variable_scope(scope_name):
        for i, size in enumerate(layer_sizes):
            inputs = tf.layers.dense(
                inputs,
                size,
                activation=tf.nn.relu if i < len(layer_sizes) - 1 else None,
                kernel_initializer=tf.contrib.layers.xavier_intializer(),
                name=scope_name + '_l' + str(i)
            )
    return inputs

batch_size = 32

states = tf.placeholder(tf.float32, shape=(batch_size, observation_size), name="state")
states_next = tf.placeholder(tf.float32, shape=(batch_size, observation_size), name="state_next")
actions = tf.placeholder(tf.int32, shape=(batch_size,), name="action")
rewards = tf.placeholder(tf.float32, shape=(batch_size,), name="reward")
done_flags = tf.placeholder(tf.float32, shape=(batch_size,), name="done")

q = dense(states, [32, 32, 2], name="Q_primary")
q_target = dense(states_next, [32, 32, 2], name="Q_target")
