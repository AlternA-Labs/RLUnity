"""
iql_unity.py
============
Unity ML‑Agents ortamında Implicit Q‑Learning (IQL) aktör‑eleştirmen eğitimi.
"""

import os, copy, random, datetime
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# ─────────────────────  1) Genel ayarlar  ────────────────────────────────── #
SEED                 = 42
MAX_ENV_STEPS        = 50_000         # toplam ortam adımı
REPLAY_CAPACITY      = 200_000
BATCH_SIZE           = 256
GAMMA                = 0.99
TAU_TARGET           = 0.005
LR_ACTOR             = 3e-4
LR_CRITIC            = 3e-4
LR_VALUE             = 3e-4
EXPECTILE_TAU        = 0.7             # τ_exp
ADV_TEMPERATURE_BETA = 3.0             # β
BC_COEFF_ALPHA       = 0.005           # α
NOISE_STD            = 0.2             # keşif için Gauss gürültüsü
DEVICE               = torch.device("mps" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# ─────────────────────  2) Replay buffer  ────────────────────────────────── #
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory   = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in idx]
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)

# ─────────────────────  3) Ağ tanımları  ─────────────────────────────────── #
def mlp(in_dim, out_dim, hidden=256, out_activation=None):
    layers = [
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim)
    ]
    if out_activation: layers.append(out_activation)
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = mlp(state_dim, action_dim, out_activation=nn.Tanh())

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):                      # Q(s,a)
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

class ValueNet(nn.Module):                    # V(s)
    def __init__(self, state_dim):
        super().__init__()
        self.net = mlp(state_dim, 1)

    def forward(self, s):
        return self.net(s)

# ─────────────────────  4) IQL özel loss fonksiyonları  ──────────────────── #
def expectile_loss(v, target, tau=EXPECTILE_TAU):
    diff = target - v
    w = torch.where(diff > 0, tau, 1 - tau)
    return (w * diff.pow(2)).mean()

def advantage_weights(q, v, beta=ADV_TEMPERATURE_BETA):
    adv = (q - v).detach()
    w   = torch.exp( beta * adv ).clamp(max=100.0)
    return w

# ─────────────────────  5) Ortam  ────────────────────────────────────────── #
print("‣ Unity ortamı açılıyor…")
env = UnityEnvironment(file_name=None, base_port=5004)
env.reset()
print('Unity environment ready!')
behavior_name = list(env.behavior_specs.keys())[0]
spec          = env.behavior_specs[behavior_name]
state_dim     = int(np.prod(spec.observation_specs[0].shape))
action_dim    = spec.action_spec.continuous_size
print(f"State dim = {state_dim},  Action dim = {action_dim}")

# ─────────────────────  6) Ağlar, optimizörler, hedefler  ───────────────── #
actor         = Actor(state_dim, action_dim).to(DEVICE)
critic        = Critic(state_dim, action_dim).to(DEVICE)
critic_target = copy.deepcopy(critic).eval().to(DEVICE)
value_net     = ValueNet(state_dim).to(DEVICE)

opt_actor  = optim.Adam(actor.parameters(),  LR_ACTOR)
opt_critic = optim.Adam(critic.parameters(), LR_CRITIC)
opt_value  = optim.Adam(value_net.parameters(), LR_VALUE)

replay = ReplayBuffer(REPLAY_CAPACITY)

# ─────────────────────  7) Eğitim döngüsü  ──────────────────────────────── #
step, ep_reward, ep_len = 0, 0.0, 0
metrics = []

decision_steps, term_steps = env.get_steps(behavior_name)
agent_id = list(decision_steps)[0]  # tek ajan varsayımı
state    = decision_steps[agent_id].obs[0].flatten()

while step < MAX_ENV_STEPS:
    # --- politika + gürültü
    with torch.no_grad():
        act = actor(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    act += np.random.normal(0, NOISE_STD, size=action_dim)
    act  = np.clip(act, -1.0, 1.0)

    env.set_actions(behavior_name, ActionTuple(continuous=act[np.newaxis, :]))
    env.step();  step += 1;  ep_len += 1

    next_decision, next_term = env.get_steps(behavior_name)
    if agent_id in next_term:
        ts   = next_term[agent_id]
        done = True
    else:
        ts   = next_decision[agent_id]
        done = False

    reward      = ts.reward
    next_state  = ts.obs[0].flatten()

    replay.push(state, act, reward, next_state, done)

    state       = next_state
    ep_reward  += reward

    # --- öğrenme
    if len(replay) >= BATCH_SIZE:
        batch = replay.sample(BATCH_SIZE)
        s  = torch.FloatTensor(np.array(batch.state)).to(DEVICE)
        a  = torch.FloatTensor(np.array(batch.action)).to(DEVICE)
        r  = torch.FloatTensor(batch.reward).unsqueeze(1).to(DEVICE)
        ns = torch.FloatTensor(np.array(batch.next_state)).to(DEVICE)
        d  = torch.FloatTensor(batch.done).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            target_q = critic_target(ns, actor(ns))
            target_y = r + GAMMA * (1 - d) * target_q

        # 1) Critic (Q)
        q = critic(s, a)
        critic_loss = nn.functional.mse_loss(q, target_y)
        opt_critic.zero_grad(); critic_loss.backward(); opt_critic.step()

        # 2) Value (V) – expectile
        with torch.no_grad():
            q_pi = critic(s, actor(s))
        v      = value_net(s)
        v_loss = expectile_loss(v, q_pi, EXPECTILE_TAU)
        opt_value.zero_grad(); v_loss.backward(); opt_value.step()

        # 3) Actor π – avantaj‑ağırlıklı BC
        with torch.no_grad():
            v_detach = value_net(s)
            q_detach = critic(s, a)
            w        = advantage_weights(q_detach, v_detach, ADV_TEMPERATURE_BETA)

        logit = actor(s)
        actor_loss = ( (logit - a).pow(2).sum(dim=1, keepdim=True) * w ).mean() * BC_COEFF_ALPHA
        opt_actor.zero_grad(); actor_loss.backward(); opt_actor.step()

        # 4) Target soft‑update
        for tp, p in zip(critic_target.parameters(), critic.parameters()):
            tp.data.mul_(1 - TAU_TARGET).add_(TAU_TARGET * p.data)

    # --- Episode bitti mi?
    if done:
        print(f"[{step:07d}]  Return = {ep_reward:.2f}  Len = {ep_len}")
        metrics.append({"step": step, "return": ep_reward})

        ep_reward, ep_len = 0.0, 0
        env.reset()
        decision_steps, _ = env.get_steps(behavior_name)
        state = decision_steps[agent_id].obs[0].flatten()

# ─────────────────────  8) Temiz kapanış  ───────────────────────────────── #
env.close()
df = pd.DataFrame(metrics)
df.to_csv("iql_training_curve.csv", index=False)
print("Eğitim tamamlandı, metrikler kaydedildi.")
