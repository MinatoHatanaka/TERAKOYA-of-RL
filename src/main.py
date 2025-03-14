import gymnasium as gym

env = gym.make("CartPole-v1", render_mode='human')
obs, info = env.reset()

print("観測データ:", obs)

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()