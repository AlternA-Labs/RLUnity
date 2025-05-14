import os
import datetime
from collections import deque, namedtuple
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.base_env import ActionTuple

# ---------------------------------------------------------------------------
# Hyper‑parameters
# ---------------------------------------------------------------------------
MAX_STEPS: int = 999_000          # Total environment steps (across all agents)
BATCH_SIZE: int = 256             # Mini‑batch size for updates
GAMMA: float = 0.99               # Discount factor
LEARNING_RATE: float = 3e-5       # Optimizer learning rate
REPLAY_BUFFER_CAPACITY: int = 100_000
TAU: float = 0.005                # Soft‑update coefficient for target nets
NOISE_STD: float = 0.15           # Initial standard deviation of exploration noise

DEVICE = torch.device("mps")  # Apple Silicon; replace with "cuda" / "cpu" as needed

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
metrics_log: List[Dict] = []      # Accumulates periodic metric snapshots
episode_lengths: List[int] = []   # Length (in steps) of completed episodes
agent_step_counts: Dict[int, int] = {}  # Per‑agent step counters within current episode

best_avg_reward: float = -float("inf")
DROP_TOLERANCE: float = 0.8       # Save checkpoint if reward drops >20 %

os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

class ReplayBuffer:
    """Fixed‑size buffer for experience tuples."""

    def __init__(self, capacity: int):
        self.memory: deque = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        """Randomly sample a batch of experiences."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------
istanbul_tz = pytz.timezone("Europe/Istanbul")
now = datetime.datetime.now(istanbul_tz)
notformatted_datetime = now.strftime("%Y-%m-%d %H:%M")
formatted_datetime = now.strftime("%Y-%m-%d_%H-%M")

# ---------------------------------------------------------------------------
# Plotting function (post‑training)
# ---------------------------------------------------------------------------

def plot_metrics(log: List[Dict]):
    """Visualise training metrics after completion."""
    df = pd.DataFrame(log)

    # Average reward with ±1 σ envelope
    plt.figure(figsize=(12, 6))
    plt.plot(df["global_step"], df["avg_reward"], label="Avg Reward")
    plt.fill_between(
        df["global_step"],
        df["avg_reward"] - df["reward_std"],
        df["avg_reward"] + df["reward_std"],
        alpha=0.2,
        label="±1σ Reward",
    )
    plt.xlabel("Global Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    # Actor / critic losses
    plt.figure(figsize=(12, 6))
    plt.plot(df["global_step"], df["actor_loss"], label="Actor Loss")
    plt.plot(df["global_step"], df["critic_loss"], label="Critic Loss")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Episode length & noise standard deviation
    plt.figure(figsize=(12, 6))
    plt.plot(df["global_step"], df["avg_ep_len"], label="Avg Episode Length")
    plt.plot(df["global_step"], df["noise_std"], label="Noise σ")
    plt.xlabel("Global Step")
    plt.grid(True)
    plt.legend()

    plt.show()


# ---------------------------------------------------------------------------
# Neural network definitions
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    """Deterministic policy network (outputs actions in [‑1, 1])."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),  # Ensures action range is [‑1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class Critic(nn.Module):
    """State‑action value network Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Utility: soft parameter update
# ---------------------------------------------------------------------------

def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """θ_target ← τ · θ_source + (1 ‑ τ) · θ_target"""
    for tgt_param, src_param in zip(target.parameters(), source.parameters()):
        tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)


# ---------------------------------------------------------------------------
# Environment initialisation
# ---------------------------------------------------------------------------
print("Waiting for Unity Editor …")

env_params = EnvironmentParametersChannel()
# If running in the Editor on *Play* mode use file_name=None with the editor port
env = UnityEnvironment(file_name=None, base_port=5004, side_channels=[env_params])

# Signal to the environment that training mode is enabled (if utilised on the
# Unity side).
env_params.set_float_parameter("is_training", 1.0)

env.reset()
print("Unity environment ready.")

# Retrieve behaviour / observation specifications
behavior_name = list(env.behavior_specs.keys())[0]
behavior_spec = env.behavior_specs[behavior_name]

obs_shape = behavior_spec.observation_specs[0].shape  # Assuming single tensor obs
state_dim = int(np.prod(obs_shape))
action_dim = behavior_spec.action_spec.continuous_size

print(f"Behavior           : {behavior_name}")
print(f"State dimension    : {state_dim}")
print(f"Continuous actions : {action_dim}\n")

# ---------------------------------------------------------------------------
# Agent bookkeeping (multiple concurrent agents supported)
# ---------------------------------------------------------------------------
agent_states: Dict[int, np.ndarray] = {}
agent_rewards: Dict[int, float] = {}
agent_done_flags: Dict[int, bool] = {}

# ---------------------------------------------------------------------------
# Instantiate networks, optimisers, replay buffer
# ---------------------------------------------------------------------------
actor = Actor(state_dim, action_dim).to(DEVICE)
critic = Critic(state_dim, action_dim).to(DEVICE)
actor_target = Actor(state_dim, action_dim).to(DEVICE)
critic_target = Critic(state_dim, action_dim).to(DEVICE)

actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

actor_optim = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optim = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
global_step = 0
episode_rewards: List[float] = []

# Obtain initial DecisionSteps / TerminalSteps from the environment
decision_steps, terminal_steps = env.get_steps(behavior_name)

# Initialise per‑agent trackers
for ag_id in decision_steps:
    obs = decision_steps[ag_id].obs[0].flatten()
    agent_states[ag_id] = obs
    agent_rewards[ag_id] = 0.0
    agent_done_flags[ag_id] = False
    agent_step_counts[ag_id] = 0

# ------------------------- Main interaction cycle ------------------------- #
while global_step < MAX_STEPS:
    # ------------------------------------------------------------------
    # 1. Produce an action for every agent awaiting a decision
    # ------------------------------------------------------------------
    agent_ids = list(decision_steps.agent_id_to_index.keys())
    actions_dict: Dict[int, np.ndarray] = {}

    for ag_id in agent_ids:
        state = agent_states[ag_id]
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            base_action = actor(state_tensor).cpu().numpy()[0]

        noise = np.random.normal(0.0, NOISE_STD, size=action_dim)
        action = np.clip(base_action + noise, -1.0, 1.0)
        actions_dict[ag_id] = action

    # ------------------------------------------------------------------
    # 2. Send the batch of actions to Unity
    # ------------------------------------------------------------------
    action_array = np.vstack([actions_dict[ag_id] for ag_id in agent_ids]).astype(np.float32)
    env.set_actions(behavior_name, ActionTuple(continuous=action_array))

    # ------------------------------------------------------------------
    # 3. Step the environment forward
    # ------------------------------------------------------------------
    try:
        env.step()
    except Exception as exc:
        print(f"Unity step error: {exc}")
        break

    global_step += 1

    # Increment per‑agent step counters (current episode)
    for ag_id in agent_ids:
        agent_step_counts[ag_id] += 1

    # ------------------------------------------------------------------
    # 4. Retrieve the next set of DecisionSteps / TerminalSteps
    # ------------------------------------------------------------------
    next_decision_steps, next_terminal_steps = env.get_steps(behavior_name)

    # ------------------------------ Terminal agents -------------------
    for ag_id in next_terminal_steps:
        reward = next_terminal_steps[ag_id].reward
        next_obs = next_terminal_steps[ag_id].obs[0].flatten()

        # Store transition
        replay_buffer.push(
            agent_states[ag_id],
            actions_dict[ag_id],
            reward,
            next_obs,
            True,
        )

        # Log episode statistics
        episode_lengths.append(agent_step_counts[ag_id])
        agent_step_counts[ag_id] = 0

        agent_rewards[ag_id] += reward
        print(
            f"[Step {global_step}] Agent {ag_id} finished | "
            f"Episode reward = {agent_rewards[ag_id]:.2f}"
        )
        episode_rewards.append(agent_rewards[ag_id])

        # Reset local trackers (Unity resets the agent internally)
        agent_states[ag_id] = next_obs
        agent_rewards[ag_id] = 0.0
        agent_done_flags[ag_id] = True

    # ------------------------------ Continuing agents ----------------
    for ag_id in next_decision_steps:
        reward = next_decision_steps[ag_id].reward
        next_obs = next_decision_steps[ag_id].obs[0].flatten()

        # Save transition (for agents we selected an action for)
        if ag_id in actions_dict:
            replay_buffer.push(
                agent_states[ag_id],
                actions_dict[ag_id],
                reward,
                next_obs,
                False,
            )

        # Update local trackers
        agent_states[ag_id] = next_obs
        agent_rewards[ag_id] += reward
        agent_done_flags[ag_id] = False

    # Keep latest step containers for the next loop iteration
    decision_steps, terminal_steps = next_decision_steps, next_terminal_steps

    # ------------------------------------------------------------------
    # 5. Parameter updates (once replay buffer is sufficiently populated)
    # ------------------------------------------------------------------
    if len(replay_buffer) >= BATCH_SIZE:
        transitions = replay_buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.from_numpy(np.asarray(batch.state)).float().to(DEVICE)
        action_batch = torch.from_numpy(np.asarray(batch.action)).float().to(DEVICE)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_state_batch = torch.from_numpy(np.asarray(batch.next_state)).float().to(DEVICE)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # Optional normalisation (example for specific observation indices)
        state_batch[:, 0:2] /= 10.0        # Example scaling for speed / direction
        next_state_batch[:, 0:2] /= 10.0
        state_batch[:, -5:] /= 10.0        # Raycast example
        next_state_batch[:, -5:] /= 10.0

        # --------------------- Critic update ----------------------
        with torch.no_grad():
            next_actions = actor_target(next_state_batch)
            target_q = critic_target(next_state_batch, next_actions)
            expected_q = reward_batch + (1 - done_batch) * GAMMA * target_q

        current_q = critic(state_batch, action_batch)
        critic_loss = nn.SmoothL1Loss()(current_q, expected_q)

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optim.step()

        # ---------------------- Actor update ----------------------
        actor_loss = -critic(state_batch, actor(state_batch)).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        # ----------------- Soft update of target nets ------------
        soft_update(actor_target, actor, TAU)
        soft_update(critic_target, critic, TAU)

    # ------------------------------------------------------------------
    # 6. Periodic logging / checkpointing
    # ------------------------------------------------------------------
    if global_step % 1_000 == 0:
        avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
        avg_reward20 = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
        reward_std = np.std(episode_rewards[-10:]) if episode_rewards else 0.0
        avg_ep_len = np.mean(episode_lengths[-10:]) if episode_lengths else 0.0
        q_mean = current_q.mean().item() if "current_q" in locals() else 0.0
        buffer_size = len(replay_buffer)
        noise_now = NOISE_STD

        print(
            f"Step {global_step:>7} | Buffer {buffer_size:>6} | "
            f"Actor Loss {actor_loss.item():.4f} | Critic Loss {critic_loss.item():.4f}"
        )

        metrics_log.append(
            {
                "global_step": global_step,
                "avg_reward": avg_reward,
                "reward_std": reward_std,
                "avg_ep_len": avg_ep_len,
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "q_mean": q_mean,
                "noise_std": noise_now,
                "buffer_size": buffer_size,
            }
        )

        # ----------------------- Checkpoint logic --------------
        if avg_reward20 > best_avg_reward:
            best_avg_reward = avg_reward20
        elif avg_reward20 < best_avg_reward * (1 - DROP_TOLERANCE):
            torch.save(actor.state_dict(), f"models/actor_drop_step{global_step}_{formatted_datetime}.pth")
            torch.save(critic.state_dict(), f"models/critic_drop_step{global_step}_{formatted_datetime}.pth")
            print(
                f"[Checkpoint] Avg reward dropped ({avg_reward20:.2f} < {best_avg_reward:.2f}). "
                "Models saved."
            )

        # ----------------------- Noise decay -------------------
        NOISE_STD = max(0.05, NOISE_STD * 0.995)

# ---------------------------------------------------------------------------
# Cleanup & final save
# ---------------------------------------------------------------------------
env.close()
print("Training complete. Shutting down Unity environment.")
try:
    plot_metrics(metrics_log)
except Exception as e:
    print(f"Error: {e}")
    training_error_occurred=True
torch.save(actor.state)
