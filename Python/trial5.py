# sac_unity.py ─ Soft Actor-Critic for the same Unity rocket task
# ───────────────────────────────────────────────────────────────
#  * Logging, plotting, checkpointing, and Unity interaction are 1-to-1
#    with your previous script.
#  * Only the RL core is replaced by SAC (twin Q-nets + stochastic actor).
#  * Works on macOS with Apple Silicon (“mps”) or switch to "cuda"/"cpu".

import os, datetime, random, pytz, torch, numpy as np, pandas as pd
from collections import deque, namedtuple
from typing import Dict, List
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple

# ─────────────────────── Hyper-parameters ──────────────────────
MAX_STEPS           = 999_000
BATCH_SIZE          = 256
GAMMA               = 0.99
LR_ACTOR            = 3e-4
LR_CRITIC           = 3e-4
LR_ALPHA            = 3e-4
REPLAY_CAPACITY     = 100_000
TAU                 = 0.005
TARGET_ENTROPY      = None        # set later = −action_dim
DEVICE              = torch.device("mps")  # Apple Silicon

training_error_occurred=False
# ────────────────────────── Logging ────────────────────────────
metrics_log, episode_lengths = [], []
best_avg_reward, DROP_TOL    = -float("inf"), 0.8
os.makedirs("models", exist_ok=True)
os.makedirs("final_models", exist_ok=True)

Transition = namedtuple("T", ("s","a","r","s2","d"))
class ReplayBuffer:
    def __init__(self,c): self.mem = deque(maxlen=c)
    def push(self,*t):     self.mem.append(Transition(*t))
    def sample(self,n):    return random.sample(self.mem,n)
    def __len__(self):     return len(self.mem)

# ──────────────────── Timestamp helpers ───────────────────────
now = datetime.datetime.now(pytz.timezone("Europe/Istanbul"))
timestr = now.strftime("%Y-m-d_%H-%M")

# ───────────────────── Plot helper (unchanged) ─────────────────
def plot_metrics(log:List[Dict]):
    df = pd.DataFrame(log)
    plt.figure(figsize=(12,6))
    plt.plot(df["step"],df["avg_r"]); plt.fill_between(df["step"],
        df["avg_r"]-df["r_std"],df["avg_r"]+df["r_std"],alpha=.2)
    plt.xlabel("Global Step"); plt.ylabel("Reward"); plt.grid(); plt.legend(["Avg ±1σ"])
    plt.figure(figsize=(12,6))
    plt.plot(df["step"],df["actor_l"]); plt.plot(df["step"],df["critic_l"])
    plt.xlabel("Global Step"); plt.ylabel("Loss"); plt.grid(); plt.legend(["Actor","Critic"])
    plt.figure(figsize=(12,6))
    plt.plot(df["step"],df["avg_len"]); plt.plot(df["step"],df["alpha"])
    plt.xlabel("Global Step"); plt.grid(); plt.legend(["Ep len","α"])
    plt.show()

# ───────────────────── Network definitions ────────────────────
class Actor(nn.Module):
    def __init__(self,sd,ad):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd,512),nn.LayerNorm(512),nn.ReLU(),
                                 nn.Linear(512,512),nn.LayerNorm(512),nn.ReLU(),
                                 nn.Linear(512,256),nn.LayerNorm(256),nn.ReLU())
        self.mu  = nn.Linear(256,ad)
        self.log = nn.Linear(256,ad)
    def forward(self,x):
        h = self.net(x)
        mu, log_std = self.mu(h), self.log(h).clamp(-20,2)
        std = log_std.exp()
        dist = Normal(mu,std)
        z    = dist.rsample()
        a_t  = torch.tanh(z)
        logp = dist.log_prob(z) - torch.log(1-a_t.pow(2)+1e-6)
        return a_t, logp.sum(1,keepdim=True)

class Critic(nn.Module):
    def __init__(self,sd,ad):
        super().__init__()
        self.q1 = self._build(sd,ad); self.q2 = self._build(sd,ad)
    @staticmethod
    def _build(sd,ad):
        return nn.Sequential(nn.Linear(sd+ad,512),nn.LayerNorm(512),nn.ReLU(),
                             nn.Linear(512,512),nn.LayerNorm(512),nn.ReLU(),
                             nn.Linear(512,256),nn.LayerNorm(256),nn.ReLU(),
                             nn.Linear(256,1))
    def forward(self,s,a):
        sa = torch.cat([s,a],1)
        return self.q1(sa), self.q2(sa)

# ───────────────────────── Environment ─────────────────────────
env_ch = EnvironmentParametersChannel()
print("Waiting for Unity to start…")
env = UnityEnvironment(file_name=None, base_port=5004, side_channels=[env_ch])

env_ch.set_float_parameter("is_training",1.0)
env.reset(); beh = list(env.behavior_specs.keys())[0]; spec = env.behavior_specs[beh]
state_dim, action_dim = int(np.prod(spec.observation_specs[0].shape)), spec.action_spec.continuous_size
TARGET_ENTROPY = -action_dim
print(f"State dim {state_dim} | Action dim {action_dim}")

# ───────────── Instantiate networks, α (entropy coeff) ─────────
actor      = Actor(state_dim,action_dim).to(DEVICE)
critic     = Critic(state_dim,action_dim).to(DEVICE)
critic_tgt = Critic(state_dim,action_dim).to(DEVICE); critic_tgt.load_state_dict(critic.state_dict())
opt_actor  = optim.Adam(actor.parameters(),LR_ACTOR)
opt_critic = optim.Adam(critic.parameters(),LR_CRITIC)
log_alpha  = torch.zeros(1,requires_grad=True,device=DEVICE)
opt_alpha  = optim.Adam([log_alpha],LR_ALPHA)
replay     = ReplayBuffer(REPLAY_CAPACITY)
α = lambda: log_alpha.exp()

# ────────────────────── Agent bookkeeping ─────────────────────
agent_states, agent_rewards, agent_steps = {}, {}, {}
global_step, episode_rewards = 0, []

dec, term = env.get_steps(beh)
for ag in dec:
    st = dec[ag].obs[0].flatten()
    agent_states[ag]  = st
    agent_rewards[ag] = 0.0
    agent_steps[ag]   = 0

# ───────────────────────── Training loop ───────────────────────
try:
    while global_step < MAX_STEPS:
        # action selection
        ids = list(dec.agent_id_to_index.keys())
        act_out = {}
        for ag in ids:
            s = torch.from_numpy(agent_states[ag]).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad(): a, _ = actor(s)
            act_out[ag] = a.cpu().numpy()[0]
        env.set_actions(beh, ActionTuple(continuous=np.vstack([act_out[ag] for ag in ids]).astype(np.float32)))

        env.step()

        global_step += 1
        for ag in ids: agent_steps[ag] += 1
        dec_next, term_next = env.get_steps(beh)

        # store transitions
        for ag in term_next:
            r = term_next[ag].reward;
            s2 = term_next[ag].obs[0].flatten()
            print(f"[Step {global_step}] Agent {ag} done. "
                  f"Episode Reward = {agent_rewards[ag] + r:.2f}")
            replay.push(agent_states[ag], act_out[ag], r, s2, True)
            episode_rewards.append(agent_rewards[ag] + r);
            episode_lengths.append(agent_steps[ag])
            agent_states[ag] = s2;
            agent_rewards[ag] = 0.0;
            agent_steps[ag] = 0
        for ag in dec_next:
            r = dec_next[ag].reward;
            s2 = dec_next[ag].obs[0].flatten()
            replay.push(agent_states[ag], act_out.get(ag, np.zeros(action_dim)), r, s2, False)
            agent_states[ag] = s2;
            agent_rewards[ag] += r
        dec, term = dec_next, term_next

        # SAC updates
        if len(replay) >= BATCH_SIZE:
            batch = Transition(*zip(*replay.sample(BATCH_SIZE)))
            S = torch.as_tensor(batch.s, dtype=torch.float32, device=DEVICE)
            A = torch.as_tensor(batch.a, dtype=torch.float32, device=DEVICE)
            R = torch.as_tensor(batch.r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
            S2 = torch.as_tensor(batch.s2, dtype=torch.float32, device=DEVICE)
            D = torch.as_tensor(batch.d, dtype=torch.float32, device=DEVICE).unsqueeze(1)

            with torch.no_grad():
                A2, logp2 = actor(S2)
                q1_t, q2_t = critic_tgt(S2, A2)
                min_q_t = torch.min(q1_t, q2_t) - α() * logp2
                y = R + (1 - D) * GAMMA * min_q_t

            q1, q2 = critic(S, A)
            crit_loss = nn.MSELoss()(q1, y) + nn.MSELoss()(q2, y)
            opt_critic.zero_grad();
            crit_loss.backward();
            opt_critic.step()

            A_samp, logp = actor(S)
            q1_a, q2_a = critic(S, A_samp)
            act_loss = (α() * logp - torch.min(q1_a, q2_a)).mean()
            opt_actor.zero_grad();
            act_loss.backward();
            opt_actor.step()

            alpha_loss = (-log_alpha * (logp + TARGET_ENTROPY).detach()).mean()
            opt_alpha.zero_grad();
            alpha_loss.backward();
            opt_alpha.step()

            with torch.no_grad():
                for p_t, p in zip(critic_tgt.parameters(), critic.parameters()):
                    p_t.data.mul_(1 - TAU).add_(TAU * p)

        # ── periodic log / ckpt every 1k steps ──
        if global_step % 1_000 == 0:
            avg_r = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            avg20 = np.mean(episode_rewards[-20:]) if episode_rewards else 0
            std_r = np.std(episode_rewards[-10:]) if episode_rewards else 0
            avg_l = np.mean(episode_lengths[-10:]) if episode_lengths else 0
            metrics_log.append(dict(step=global_step, avg_r=avg_r, r_std=std_r,
                                    avg_len=avg_l, actor_l=act_loss.item(),
                                    critic_l=crit_loss.item(), alpha=α().item()))
            # replace the whole print block
            alpha_val = α().item()
            print(f"Step {global_step:7} | Buf {len(replay):6} | "
                  f"α {alpha_val:.3f} | Act {act_loss.item():.4f} | Crit {crit_loss.item():.4f}")
            if avg20 > best_avg_reward:
                best_avg_reward = avg20
            elif avg20 < best_avg_reward * (1 - DROP_TOL):
                torch.save(actor.state_dict(), f"models/actor_drop_{global_step}_{timestr}.pth")
                torch.save(critic.state_dict(), f"models/critic_drop_{global_step}_{timestr}.pth")
                print("[Checkpoint] Reward drop – models saved.")
except Exception as er:
    print(f"Error: {er}")
finally:
    #   ↓ yalnızca bir kez çalışır; Unity kapanmış olsa bile güvenlidir
    try:
        env.close()
    except Exception:
        pass

    # metrikler hiç oluşmadıysa korumalı değerler
    dummy = lambda: 0.0
    last_act_loss = act_loss.item() if "act_loss" in locals() else dummy()
    last_critic_loss = crit_loss.item() if "crit_loss" in locals() else dummy()

    torch.save(actor.state_dict(),
               f"final_models/actor_final_{timestr}.pth")
    torch.save(critic.state_dict(),
               f"final_models/critic_final_{timestr}.pth")
    print("Model Saved.")

    try:
        plot_metrics(metrics_log)
    except Exception as e:
        print(f"Plot çizilemedi: {e}")
