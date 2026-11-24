import gymnasium as gym 

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# Create an environment for LunarLander-v3
def main():
    train_env = make_vec_env(
        "LunarLander-v3",
        n_envs=4,
        vec_env_cls=SubprocVecEnv
    )
                             
    # Define the model (PPO)
    model = DQN(policy='MlpPolicy',
            env=train_env,
            learning_rate=6.3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=128,
            train_freq=4,
            gradient_steps=-1,
            target_update_interval=250,
            exploration_fraction=0.15,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            policy_kwargs=dict(net_arch=[256, 256]),
            gamma=0.99,
            device="cpu",
            verbose=1
            )

    # Train the model for 1,500,000 timesteps
    print("Starting training Vectorized DQN .....")
    model.learn(total_timesteps=400000, progress_bar=True)

    # Save the trained model
    model_name = "dqn-vec-lunarlander-v3" 
    model.save("models/dqn-vec-lunarlander-v3")
    print(f"Model saved as dqn-vec-lunarlander-v3.zip in ./videos/")
    train_env.close()

    # Evaluate the model
    # Create a new environment for evaluation
    eval_env = DummyVecEnv([lambda: Monitor(gym.make("LunarLander-v3", render_mode='rgb_array'))])

    # Evaluate the model for 10 evaluation episodes and deterministic=True
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    eval_env.close()

    # print te results
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

if __name__ == "__main__":
    main()


