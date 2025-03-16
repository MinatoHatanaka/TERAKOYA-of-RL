import gymnasium as gym

env = gym.make("CartPole-v1", render_mode='human')
obs, info = env.reset()

print("観測データ:", obs)

episode_over = False
while not episode_over:
    #action = env.action_space.sample()
    cart_pos, cart_vel, pole_angle, pole_vel = obs
    action = 1 if pole_angle > 0 else 0
    obs, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()