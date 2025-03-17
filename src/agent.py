from collections import defaultdict
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class Agent:
    def __init__(self, env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs):
        raise NotImplementedError

    def update(self, obs, action, reward, terminated, next_obs):
        raise NotImplementedError

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

class CartPoleAgent(Agent):
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay:float,
            discount_factor: float = 0.95,
    ):
        super().__init__(env, learning_rate, initial_epsilon, epsilon_decay, discount_factor)

    def get_action(self, obs: Box(low=[-4.8, -np.inf, -0.41887903, -np.inf], high=[4.8, np.inf, 0.41887903, np.inf], shape=(4,), dtype=np.float32)) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self,
               obs: Box(low=[-4.8, -np.inf, -0.41887903, -np.inf], high=[4.8, np.inf, 0.41887903, np.inf], shape=(4,), dtype=np.float32),
               action: int,
               reward: float,
               terminated: bool,
               next_obs: Box(low=[-4.8, -np.inf, -0.41887903, -np.inf], high=[4.8, np.inf, 0.41887903, np.inf], shape=(4,), dtype=np.float32),
    ):
        future_q_values = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_values - self.q_values[obs][action]
        )
        self.training_error.append(temporal_difference)