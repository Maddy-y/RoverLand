import os 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

#-------------------Configuration---------------------
num_runs = 10
eval_episodes = 15
ppo_timesteps = 1500000
dqn_timesteps = 400000

MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Create an environment for LunarLander-v3
def train_eval_ppo(seed: int):
    train_env = make_vec_env(
        "LunarLander-v3",
        n_envs=8,
        seed=seed,
        vec_env_cls=SubprocVecEnv
    )
                             
    # Define the model (PPO)
    model = PPO(policy='MlpPolicy',
            env=train_env,
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.005,
            vf_coef=0.5,
            device="auto",
            verbose=1
            )

    # Train the model for 1,500,000 timesteps
    print("Starting training.....")
    model.learn(total_timesteps=ppo_timesteps, progress_bar=True)
    train_env.close()

    # Evaluate the model
    # Create a new environment for evaluation
    eval_env = Monitor(gym.make("LunarLander-v3", render_mode='rgb_array'))
    eval_env.reset(seed=seed + 1000)

    # Evaluate the model for 10 evaluation episodes and deterministic=True
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)
    eval_env.close()

    return model, mean_reward, std_reward

# Create an environment for LunarLander-v3
def train_eval_dqn(seed: int):
    train_env = make_vec_env(
        "LunarLander-v3",
        n_envs=4,
        seed=seed,
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
            device="auto",
            verbose=1
            )

    # Train the model for 1,500,000 timesteps
    print("Starting training Vectorized DQN .....")
    model.learn(total_timesteps=dqn_timesteps, progress_bar=True)
    train_env.close()

    # Evaluate the model
    # Create a new environment for evaluation
    eval_env = DummyVecEnv([lambda: Monitor(gym.make("LunarLander-v3", render_mode='rgb_array'))])
    eval_env.seed(seed + 1000)
    eval_env.reset()

    # Evaluate the model for 10 evaluation episodes and deterministic=True
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)
    eval_env.close()

    return model, mean_reward, std_reward

def plot_benchmark_results(ppo_scores, dqn_scores):
    """Generate a comparison graph of all runs with mean lines and solved threshold"""
    runs = np.arange(1, num_runs + 1)
    
    plt.figure(figsize=(12, 6), dpi=150)
    
    # Plot individual run scores
    plt.plot(runs, ppo_scores, marker='o', linewidth=2.5, color='#2563EB', label=f'PPO (Avg: {np.mean(ppo_scores):.1f})')
    plt.plot(runs, dqn_scores, marker='s', linewidth=2.5, color='#DC2626', label=f'DQN (Avg: {np.mean(dqn_scores):.1f})')
    
    # 200 points "Solved" threshold line
    plt.axhline(y=200, color='#16A34A', linestyle='--', linewidth=2, label='Solved Threshold (+200)')
    
    # Formatting
    plt.title("LunarLander-v3: PPO vs DQN Multi-Run Benchmark (10 Seeds, Unwrapped)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Run Index (Seed)", fontsize=12)
    plt.ylabel(f"Evaluation Mean Reward ({eval_episodes} Episodes)", fontsize=12)
    plt.xticks(runs)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    
    plot_file = PLOTS_DIR / "ppo_vs_dqn_10_runs_benchmark.png"
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"\nBenchmark visualization saved to: {plot_file}")
    plt.show()

def main():
    print("=" * 70)
    print(f"Starting 10-Run Benchmark for PPO and DQN on LunarLander-v3")
    print(f"Evaluation: {eval_episodes} episodes per run | Solved Criterion: >= 200.0")
    print("=" * 70)
    
    ppo_scores = []
    dqn_scores = []
    
    best_ppo_score = -float("inf")
    best_dqn_score = -float("inf")
    
    # ----------------- 1. Train PPO (10 runs) -----------------
    print("\n--- Benchmarking PPO (10 Runs) ---")
    for i in range(1, num_runs + 1):
        seed = i * 42
        print(f"[PPO Run {i:02d}/{num_runs}] Training with seed {seed}...", end=" ", flush=True)
        model, mean_rew, std_rew = train_eval_ppo(seed)
        ppo_scores.append(mean_rew)
        print(f"Mean Score: {mean_rew:6.2f} +/- {std_rew:5.2f}")
        
        # Save highest scoring model
        if mean_rew > best_ppo_score:
            best_ppo_score = mean_rew
            best_path = MODELS_DIR / "best_ppo_model"
            model.save(best_path)
            print(f"  --> [NEW BEST PPO] Saved to {best_path}.zip ({best_ppo_score:.2f})")
            
    # ----------------- 2. Train DQN (10 runs) -----------------
    print("\n--- Benchmarking DQN (10 Runs) ---")
    for i in range(1, num_runs + 1):
        seed = i * 42
        print(f"[DQN Run {i:02d}/{num_runs}] Training with seed {seed}...", end=" ", flush=True)
        model, mean_rew, std_rew = train_eval_dqn(seed)
        dqn_scores.append(mean_rew)
        print(f"Mean Score: {mean_rew:6.2f} +/- {std_rew:5.2f}")
        
        # Save highest scoring model
        if mean_rew > best_dqn_score:
            best_dqn_score = mean_rew
            best_path = MODELS_DIR / "best_dqn_model"
            model.save(best_path)
            print(f"  --> [NEW BEST DQN] Saved to {best_path}.zip ({best_dqn_score:.2f})")

    # ----------------- 3. Summary Statistics -----------------
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"PPO Aggregate Mean:  {np.mean(ppo_scores):6.2f} +/- {np.std(ppo_scores):5.2f} | Best: {best_ppo_score:6.2f}")
    print(f"DQN Aggregate Mean:  {np.mean(dqn_scores):6.2f} +/- {np.std(dqn_scores):5.2f} | Best: {best_dqn_score:6.2f}")
    print(f"PPO Solved Runs (>= 200): {sum(1 for s in ppo_scores if s >= 200)} / {num_runs}")
    print(f"DQN Solved Runs (>= 200): {sum(1 for s in dqn_scores if s >= 200)} / {num_runs}")
    print("=" * 70)

    # ----------------- 4. Visualization -----------------
    plot_benchmark_results(ppo_scores, dqn_scores)

if __name__ == "__main__":
    main()