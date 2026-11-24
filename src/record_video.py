import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

# Configuaration
num_eval_episode = 4
# Load the trained agenet
model = PPO.load("ppo-lunarlander-v3")

# Setup the base environment to run the trained agent
base_env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(
    base_env,
    video_folder="videos",
    name_prefix="eval",
    episode_trigger=lambda x: True 
)

# Add episodes statistics tracking
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episode)

print(f"Starting evaluation for {num_eval_episode} episodes")
print(f"Videos will be saved to lunar_replays/")

# Run 3 demonstration landings
for episode in range(num_eval_episode):
    obs, info = env.reset()
    ep_reward = 0
    step_count = 0

    episode_over = False
    while not episode_over:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        step_count += 1


        episode_over = terminated or truncated

    print(f"Episode {episode + 1}: {step_count} step, reward = {ep_reward}")

env.close()
print("Videos saved to ./videos/")

# Print summary statistics
print(f'\nEvaluation Summary:')
print(f'Episode durations: {list(env.time_queue)}')
print(f'Episode rewards: {list(env.return_queue)}')
print(f'Episode lengths: {list(env.length_queue)}')

# Calculate some useful metrics
avg_reward = np.mean(env.return_queue)
avg_length = np.mean(env.length_queue)
std_reward = np.std(env.return_queue)

print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
print(f'Average episode length: {avg_length:.1f} steps')
print(f'Success rate: {sum(1 for r in env.return_queue if r > 0) / len(env.return_queue):.1%}')