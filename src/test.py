import gymnasium as gym

env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=100_000)
print(env.reset())