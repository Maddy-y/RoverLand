# RoverLand: Deep Reinforcement Learning for Autonomous Lunar Landing

An end-to-end Reinforcement Learning (RL) pipeline trained on Gymnasium's `LunarLander-v3` environment using **Stable-Baselines3**, featuring custom reward shaping, policy evaluation benchmarks, and REST/containerized inference support.

An Overview:
Lunar Lander is a free game where you try to land a small rocket on a platform without crashing. It was originally part of OpenAI's Gym and is now available through Gymnasium, an open-source library.

The idea is simple: instead of controlling the rocket yourself, you train an AI to do it. The AI tries landing thousands of times, getting rewarded when it does something right and penalized when it makes mistakes. Eventually, it figures out how to land on its own.

This is called **reinforcement learning**. It's basically like training a dog with treats, except the treats are points.
 
The AI learns based on the points it earns, so what you reward it for matters. By changing how points are awarded, you can guide it toward the behavior you want. This is called **reward shaping**, and it's what you'll be experimenting with in this project.

The same general approach is used in real robotics. Tesla has said its Optimus robot learned to dance entirely through reinforcement learning in simulation.

---

## Key Features
* **Multi-Algorithm Benchmarking:** Compares Proximal Policy Optimization (PPO) and Deep Q-Networks (DQN) across identical evaluation seeds.
* **Custom Reward Shaping:** Modular `gymnasium.Wrapper` that penalizes excessive tilt, angular velocity, and fuel burn while rewarding smooth vertical descent near the landing pad.
* **Reproducible Evaluation & Logging:** Automated evaluation scripts generating reward curves, episode length metrics, and landing success rates.
* **Automated Testing & Deployment Ready:** Includes `pytest` unit tests for environment wrappers and inference endpoints structured for **FastAPI** and **Docker**.

---

## Project Structure
```bash
roverland/
├── src/
│   ├── envs/           # Custom Gymnasium reward-shaping wrappers
│   ├── train.py        # Training entrypoint for PPO and DQN agents
│   ├── evaluate.py     # Deterministic evaluation & metric aggregation
│   └── api.py          # FastAPI inference service for trained policies
├── tests/              # Unit tests for custom wrappers and action shapes
├── plots/              # Training reward curves and benchmark comparisons
├── models/             # Saved model checkpoints (ignored in Git; see Releases/Drive)
├── Dockerfile          # Containerized runtime for inference
├── requirements.txt    # Python dependencies
└── README.md
```
---

## How it works
 
1. The game hands the AI eight numbers. The game (the **environment**) runs at 50 **frames** a second (a frame is one tick of the physics) and hands the AI eight numbers each frame, the **observation**. It never sees the picture, only these:
 
| # | What it is | Example | 
|---|------------|---------|
| 1 | left / right of the pad | 0.20 (a bit to the right)|
| 2 | height above the pad | 0.90 (still high) |
| 3 | sideways speed | -0.10 (drifting left) |
| 4 | up / down speed | -0.50 (falling) |
| 5 | tilt (in radians) | 0.05 (nearly upright) |
| 6 | how fast it is spinning | 0.00 (not spinning) |
| 7 | left leg on the ground | 0 (no) |
| 8 | right leg on the ground | 0 (no) |

 
2. The AI picks one of four actions. Do nothing, fire the left side engine, fire the main engine, or fire the right side engine. One choice per frame.
 
3. The brain is a function with dials. A **neural network** here is just a function: eight numbers go in, a preference for each action comes out. In between sit a few thousand dials (**weights**), random at the start, so the rocket flails. You don't build it: **Stable-Baselines3**, a free Python library, hands you one, small enough (two layers of 64 units) for a laptop.
 
4. Every frame gets a score. This is the **reward**, and Lunar Lander already comes with one. Each frame it works out a "how good is this position" number: minus 100 times the distance from the pad, minus 100 times the speed, minus 100 times the tilt, plus 10 per leg on the ground. The frame's reward is how much that number improved since the last frame, minus 0.3 if the main engine fired or 0.03 if a side engine did (fuel). Counting version:
 
| Frame | Distance | Speed | Tilt | Legs down | Position score |
|-------|----------|-------|------|-----------|----------------|
| 1     | 0.50     | 0.40  | 0.10 |     0     |-50 - 40 - 10 = -100|
| 2     |   0.49   |  0.38 | 0.10 |     0     |-49 - 38 - 10 = -97 |

 
The score improved by 3. The main engine fired to do it, so the frame pays 3 minus 0.3, which is 2.7. On the last frame the reward is replaced by plus 100 for resting safely or minus 100 for a crash. Add every frame up and you get the score for the attempt (an **episode**).
 
5. Training is play, score, adjust. The AI flies for a while, sees which choices led to more points, and nudges the dials toward them. Then it flies again. The nudging recipe is **PPO** (Proximal Policy Optimization, from OpenAI in 2017). Its one idea is to keep each update small, so one lucky stretch can't wreck what the AI has learned.
 
6. It learns exactly what you scored. Say you add a rule: plus 1 for every frame the rocket is upright and slow. An attempt is cut off at 1,000 frames, so a rocket that just hangs in the air can collect up to 1,000 points from your rule and pay at most 300 in fuel. Landing ends the attempt and the payments stop, so hovering beats landing, and training can learn exactly that. This is called **reward hacking**. The fix is in the reward: pay for getting better (the change between frames, like the built-in score does) instead of for sitting in a nice spot, and add a small fee every frame so waiting costs something.

## Tech Stack
- RL & Simulation: Python, Gymnasium (**Box2D**), Stable-Baselines3, PyTorch
- Analysis & Visualization: NumPy, Matplotlib
- Testing & Serving: Pytest, FastAPI, Uvicorn, Docker
