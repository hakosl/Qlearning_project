import tensorflow as tf
import gym
from collections import defaultdict
import numpy as np

class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """This wrapper converts a Box observation into a single integer.
    """
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)

        low = np.array(low)
        high = np.array(high)
        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        self.observation_space = gym.spaces.Discrete(n_bins ** low.flatten().shape[0])

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._convert_to_one_number(digits)



env = gym.make('CartPole-v1')
env = DiscretizedObservationWrapper(
    env, 
    n_bins=8, 
    low=[-2.4, -2.0, -0.42, -3.5], 
    high=[2.4, 2.0, 0.42, 3.5]
)


Q = defaultdict(float)

gamma = 0.99
alpha = 0.5
n_steps = 10000
epsilon = 0.1
rewards = []
reward = 0.0


ob = env.reset()
actions = range(env.action_space.n)

def update_Q(s, r, a, s_next, done):
    max_q_next = max([Q[s_next, a] for a in actions])
    
    Q[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a])
    

    



def act(ob):
    if np.random.random() < epsilon:
        return env.action_space.sample()

    
    qvals = {a: Q[ob, a] for a in range(env.action_space.n)}
    max_q = max(qvals.values())
    
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)


env.reset()


for step in range(n_steps):
    a = act(ob)
    env.unwrapped.render()
    ob_next, r, done, _ = env.step(a)
    
    update_Q(ob, r, a, ob_next, done)
    reward += r
    
    if done:
        rewards.append(reward)
        reward = 0.0
        ob = env.reset()
    else:
        ob = ob_next