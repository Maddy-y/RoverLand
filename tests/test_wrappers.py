import pytest
import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

# Import your custom wrapper from src/wrappers.py
# from src.wrappers import CustomRewardWrapper


@pytest.fixture
def base_env():
    env = gym.make("LunarLander-v3")
    yield env
    env.close()


def test_gym_api_compliance(base_env):
    """Ensures wrapper complies with Gymnasium/SB3 standards."""
    # wrapped_env = CustomRewardWrapper(base_env)
    # check_env(wrapped_env)  # SB3's built-in validator
    pass


def test_reward_shaping_logic(base_env):
    """Verify that specific conditions trigger the expected shaped reward."""
    # wrapped_env = CustomRewardWrapper(base_env)
    # obs, info = wrapped_env.reset()
    # obs, reward, term, trunc, info = wrapped_env.step(action=0)
    # assert isinstance(reward, float)
    # assert not np.isnan(reward)
    pass


def test_wrapper_reset_clears_state(base_env):
    """Ensure internal tracking state is cleared between episodes."""
    # wrapped_env = CustomRewardWrapper(base_env)
    # wrapped_env.reset()
    # wrapped_env.step(0)
    # wrapped_env.reset()
    # assert wrapped_env.prev_potential is None
    pass