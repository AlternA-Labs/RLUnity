import time, os, numpy as np, torch, torch.nn as nn
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple
import matplotlib.pyplot as plt
import pandas as pd

# ───────────────────────── User settings ─────────────────────────
MODEL_PATH ="models/actor_drop_131000_2025-m-d_12-38.pth"
UNITY_EXEC = None
BASE_PORT  = 5004
MAX_EPISODES = 20
RENDER_EVERY_STEP = True
NOISE_STD = 0.0
SUCCESS_THRESHOLD = 0.0

# ───────────────────────── Device setup ─────────────────────────
device = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

# ───────────────────────── Actor network ─────────────────────────
class Actor(nn.Module):
    """SAC stochastic actor; returns tanh‑squashed actions."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU()
        )
        self.mu  = nn.Linear(256, action_dim)
        self.log = nn.Linear(256, action_dim)

    def forward(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        h  = self.net(x)
        mu = self.mu(h)
        if deterministic:
            return torch.tanh(mu)
        log_std = self.log(h).clamp(-20, 2)
        std = log_std.exp()
        z   = torch.randn_like(mu) * std + mu
        return torch.tanh(z)

# ───────────────────────── Helper functions ─────────────────────

def preprocess(obs_batch: np.ndarray) -> torch.Tensor:
    """Apply the same normalisation used during training."""
    obs_batch = obs_batch.copy()
    obs_batch[:, 0]  /= 10.0
    obs_batch[:, 1]  /= 10.0
    obs_batch[:, -5:] /= 10.0
    return torch.as_tensor(obs_batch, dtype=torch.float32, device=device)


def act(state_np: np.ndarray) -> np.ndarray:
    state_t = preprocess(state_np)
    with torch.no_grad():
        action = actor(state_t).cpu().numpy()
    if NOISE_STD > 0:
        action += np.random.normal(0, NOISE_STD, size=action.shape)
    return np.clip(action, -1.0, 1.0)

# ───────────────────────── Launch Unity ─────────────────────────
print("Unity starting…")
params_ch = EnvironmentParametersChannel()
env = UnityEnvironment(file_name=UNITY_EXEC, base_port=BASE_PORT,
                       side_channels=[params_ch])
params_ch.set_float_parameter("is_training", 0.0)
env.reset()

behavior_name = list(env.behavior_specs.keys())[0]
spec          = env.behavior_specs[behavior_name]
state_dim     = int(np.prod(spec.observation_specs[0].shape))
action_dim    = spec.action_spec.continuous_size
print(f"Behavior: {behavior_name} | state_dim: {state_dim} | action_dim: {action_dim}")

# ───────────────────────── Load model ───────────────────────────
actor = Actor(state_dim, action_dim).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
actor.load_state_dict(state_dict)
actor.eval()
print(f"Model loaded from {MODEL_PATH}")

# ─────────────────── Episode / step bookkeeping ─────────────────
episode_returns, episode_success = [], []

cur_episode_id = 0

decision_steps, _ = env.get_steps(behavior_name)
agent_state = decision_steps[0].obs[0].flatten()
agent_return = 0.0

global_step = 0

# ───────────────────────── Inference loop ───────────────────────
while cur_episode_id < MAX_EPISODES:

    action_np = act(agent_state[np.newaxis, :])
    env.set_actions(behavior_name, ActionTuple(continuous=action_np.astype(np.float32)))

    env.step(); global_step += 1

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    if len(terminal_steps):
        term = terminal_steps[0]
        agent_return += term.reward
        print(f"Episode {cur_episode_id} finished | reward {agent_return:.2f} | steps {global_step}")


        episode_returns.append(agent_return)
        episode_success.append(agent_return >= SUCCESS_THRESHOLD)

        cur_episode_id += 1
        agent_return = 0.0
        decision_steps, _ = env.get_steps(behavior_name)
        agent_state = decision_steps[0].obs[0].flatten()
    else:

        dec = decision_steps[0]
        agent_return += dec.reward
        agent_state = dec.obs[0].flatten()

    if RENDER_EVERY_STEP:
        time.sleep(1/30)

print("Inference done, closing Unity…")
env.close()

if episode_returns:
    df = pd.DataFrame({
        "episode": range(1, len(episode_returns)+1),
        "reward" : episode_returns,
        "success": np.array(episode_success, dtype=int)
    })

    plt.figure(figsize=(10,4))
    plt.plot(df["episode"], df["reward"], marker="o")
    plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True)
    plt.title("Inference rewards (SAC)")
    plt.tight_layout()

    plt.figure(figsize=(10,4))
    cum_succ = df["success"].cumsum() / df["episode"]
    plt.plot(df["episode"], cum_succ, marker="o")
    plt.xlabel("Episode"); plt.ylabel("Cumulative success rate")
    plt.ylim(0,1); plt.grid(True); plt.title("Success ratio")
    plt.tight_layout()
    plt.show()
else:
    print("No episodes completed – skipping plots.")
    print(f"Average reward: {np.mean(episode_returns):.2f}")
    print(f"Success %: {np.mean(episode_success) * 100:.2f}%")