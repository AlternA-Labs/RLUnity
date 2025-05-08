import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import datetime
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple
import matplotlib.pyplot as plt
import pytz
import pandas as pd


##########################################
# Hiperparametreler
##########################################
MAX_STEPS = 999_000             # Toplam adım sayısı (tüm ajanlar için toplu)
BATCH_SIZE = 256                 # Mini-batch boyutu
GAMMA = 0.99                    # İndirim faktörü
LEARNING_RATE = 3e-5            # Öğrenme hızı
REPLAY_BUFFER_CAPACITY = 100_000
TAU = 0.005                     # Soft update katsayısı

# Exploration için Gaussian gürültü
NOISE_STD = 0.15

device = torch.device("mps" )#if torch.mps.is_available() else "cpu")
metrics_log = []

best_avg_reward = -float("inf")   # Şu ana kadarki en yüksek ortalama ödül
DROP_TOLERANCE = 0.8
os.makedirs("models", exist_ok=True)
##########################################
# Replay Buffer
##########################################
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



#dates
utc_plus_3 = pytz.timezone('Europe/Istanbul')
date = datetime.datetime.now(utc_plus_3)
notformatted_datetime=date.strftime("%Y-%m-%d %H:%M")
formatted_datetime = date.strftime("%Y-%m-%d_%H:%M")

training_error_occurred=False

def plot(metrics_log1):
    metrics_df1 = pd.DataFrame(metrics_log1)
    # grafik
    plt.figure(figsize=(12, 10))
    plt.plot(metrics_df1["global_step"], metrics_df1["actor_loss"], color="orange", label="Actor Loss")
    plt.plot(metrics_df1["global_step"], metrics_df1["critic_loss"], color="blue", label="Critic Loss")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Actor ve Critic Loss vs Global Step")
    plt.figtext(0.95, 0.01, f'{notformatted_datetime}',
                ha='right', va='bottom', fontsize=10, color='gray')
    plt.legend()
    plt.grid(True)
    plt.figure(figsize=(12, 10))
    plt.plot(metrics_df1["global_step"], metrics_df1["avg_reward"], color="pink", label="Avearge Reward")
    plt.xlabel("Global Step")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Global Step")
    plt.figtext(0.95, 0.01, f'{notformatted_datetime}',
                ha='right', va='bottom', fontsize=10, color='gray')
    plt.show()

# Actor (Policy) Ağı
##########################################
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
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
            nn.Tanh()            # output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

##########################################
# Critic (Q-value) Ağı
##########################################
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
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
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

##########################################
# Ortamı (Tek Environment) Başlatma
##########################################
print('waiting unity')
# Editor'de Play mode'da: file_name=None, base_port=5004 (Project Settings/ML-Agents -> Editor Port)
env_params = EnvironmentParametersChannel()
env = UnityEnvironment(file_name=None, base_port=5004, side_channels=[env_params])
env_params.set_float_parameter("is_training", 1.0)#Training
env.reset()
print('ekin')
# Mevcut Behavior adlarını al
behavior_names = list(env.behavior_specs.keys())
behavior_name = behavior_names[0]
spec = env.behavior_specs[behavior_name]

# Observation/Action boyutlarını çıkar
obs_shape = spec.observation_specs[0].shape  # İlk gözlem tensoru
state_dim = int(np.prod(obs_shape))          # Flatten boyutu
action_dim = spec.action_spec.continuous_size

print(f"Behavior: {behavior_name}")
print(f"State dim: {state_dim}, Action dim: {action_dim}")

##########################################
# Ajan Durumlarını Takip
##########################################
# Birden çok agent olacak, her agent_id için son state'i saklamalıyız.
agent_states = {}    # agent_id -> en son gözlem
agent_rewards = {}   # agent_id -> birikmiş ödül (loglamak için)
agent_done_flags = {}# agent_id -> done mı

##########################################
# Ağların, Optimizer'ların ve Replay Buffer'ın Oluşturulması
##########################################
actor = Actor(state_dim, action_dim).to(device)
critic = Critic(state_dim, action_dim).to(device)
actor_target = Actor(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)

actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

def soft_update(target_net, source_net, tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

##########################################
# Eğitim Döngüsü
##########################################
global_step = 0
episode_rewards = []
# İlk adım verilerini çekiyoruz
decision_steps, terminal_steps = env.get_steps(behavior_name)

# decision_steps içindeki her agent_id'nin ilk gözlemini sakla
for agent_id in decision_steps:
    obs = decision_steps[agent_id].obs[0].flatten()
    agent_states[agent_id] = obs
    agent_rewards[agent_id] = 0.0
    agent_done_flags[agent_id] = False

while global_step < MAX_STEPS:
    # 1) Karar bekleyen tüm agent'lar için aksiyon oluştur
    agent_ids = list(decision_steps.agent_id_to_index.keys())
    actions_dict = {}  # agent_id -> aksiyon

    for agent_id in agent_ids:
        state = agent_states[agent_id]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            base_action = actor(state_tensor).cpu().numpy()[0]

        noise = np.random.normal(0, NOISE_STD, size=action_dim)
        action = base_action + noise
        action = np.clip(action, -1.0, 1.0)

        actions_dict[agent_id] = action

    # 2) Aksiyonları environment'a gönderelim
    # set_actions, agent_id -> index sıralı array istediği için:
    # Agent'ların sırası agent_ids'teki gibidir.
    # Bu sıralama ile bir (N, action_dim) dizisi oluşturuyoruz
    # N = len(agent_ids)
    action_array = np.zeros((len(agent_ids), action_dim), dtype=np.float32)
    for i, agent_id in enumerate(agent_ids):
        action_array[i, :] = actions_dict[agent_id]
    action_tuple = ActionTuple(continuous=action_array)
    env.set_actions(behavior_name, action_tuple)

    # 3) Ortamı bir adım ilerlet
    try:
        env.step()
    except Exception as e:
        print(f"Hata: {e}")
        break
    global_step += 1

    # 4) Yeni decision_steps ve terminal_steps al
    next_decision_steps, next_terminal_steps = env.get_steps(behavior_name)

    # 4a) Terminal olan agentlar
    for agent_id in next_terminal_steps:
        reward = next_terminal_steps[agent_id].reward
        next_obs = next_terminal_steps[agent_id].obs[0].flatten()
        done = True

        # Replay'e geçiş ekle
        # (S, A, R, S', done)
        old_state = agent_states[agent_id]
        old_action = actions_dict[agent_id]
        replay_buffer.push(old_state, old_action, reward, next_obs, done)

        # Log
        agent_rewards[agent_id] += reward
        print(f"[Step {global_step}] Agent {agent_id} done. Episode Reward = {agent_rewards[agent_id]:.2f}")
        episode_rewards.append(agent_rewards[agent_id])
        # Agent'i sıfırlıyoruz (genelde OnEpisodeBegin unity tarafında)
        # Yine de python tarafında old state vb. güncelle:
        agent_states[agent_id] = next_obs
        agent_rewards[agent_id] = 0.0
        agent_done_flags[agent_id] = True

    # 4b) Devam eden (decision) agentlar
    for agent_id in next_decision_steps:
        reward = next_decision_steps[agent_id].reward
        next_obs = next_decision_steps[agent_id].obs[0].flatten()
        done = False

        # Replay'e geçiş ekle
        if agent_id in actions_dict:  # Yalnızca bu adımdan önce aksiyon ürettiğimiz agent
            old_state = agent_states[agent_id]
            old_action = actions_dict[agent_id]
            replay_buffer.push(old_state, old_action, reward, next_obs, done)

        # Agent state güncelle
        agent_states[agent_id] = next_obs
        agent_rewards[agent_id] += reward
        agent_done_flags[agent_id] = False

    # Son olarak güncel decision_steps/terminal_steps referanslarını sakla
    decision_steps, terminal_steps = next_decision_steps, next_terminal_steps

    # 5) Replay Buffer yeterliyse (>= BATCH_SIZE) ağları güncelle
    if len(replay_buffer) >= BATCH_SIZE:
        transitions = replay_buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.FloatTensor(np.array(batch.action)).to(device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(device)

        # ---- Normalize observations ----
        # Örneğin: dir.magnitude ve fwdSpeed için max 10, raycast için max 10
        state_batch[:, 0] /= 10.0    # dir.magnitude
        state_batch[:, 1] /= 10.0    # fwdSpeed
        state_batch[:, -5:] /= 10.0  # son 5 raycast gözlemi
        next_state_batch[:, 0] /= 10.0
        next_state_batch[:, 1] /= 10.0
        next_state_batch[:, -5:] /= 10.0
        # Critic güncelleme
        with torch.no_grad():
            next_actions = actor_target(next_state_batch)
            target_q = critic_target(next_state_batch, next_actions)
            expected_q = reward_batch + (1 - done_batch) * GAMMA * target_q

        #current_q = critic(state_batch, action_batch)
        #critic_loss = nn.MSELoss()(current_q, expected_q)

        #critic_optimizer.zero_grad()
        #critic_loss.backward()
        #critic_optimizer.step()

        # ---- Huber loss ----
        current_q = critic(state_batch, action_batch)
        # ---- Huber loss ----
        critic_loss = nn.SmoothL1Loss()(current_q, expected_q)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        # ---- Gradient clipping ----
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()
        # Actor güncelleme
        actor_loss = -critic(state_batch, actor(state_batch)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Target network soft update
        soft_update(actor_target, actor, TAU)
        soft_update(critic_target, critic, TAU)

    # İlerleme logu
# İlerleme logu
    if global_step % 1000 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        avg_reward2 = np.mean(episode_rewards[-20:])


        print(f"Step: {global_step}, Replay Buffer: {len(replay_buffer)}, Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
        metrics_log.append({
            "global_step": global_step,
            "avg_reward": avg_reward,
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        })

        if avg_reward2 > best_avg_reward:
            best_avg_reward = avg_reward2

        elif avg_reward2 < best_avg_reward * (1 - DROP_TOLERANCE):
            torch.save(actor.state_dict(),
                       f"models/actor_drop_step{global_step}_{formatted_datetime}.pth")
            torch.save(critic.state_dict(),
                       f"models/critic_drop_step{global_step}_{formatted_datetime}.pth")
            print(f"[{global_step}] Ortalama ödül düştü ({avg_reward:.2f} < "
                  f"{best_avg_reward:.2f}). Modeller kaydedildi.")
            print(f"model ismi: models/actor_{global_step}_{formatted_datetime}")


        # ---- Noise decay ----
        NOISE_STD = max(0.05, NOISE_STD * 0.995)

# Eğitim bitince ortamı kapat

env.close()
print("Eğitim tamamlandı.")

torch.save(actor.state_dict(), f"models/actor{formatted_datetime}.pth")
torch.save(critic.state_dict(), f"models/critic{formatted_datetime}.pth")
print(f"Son adımda Actor ve Critic modeli kaydedildi: actor{formatted_datetime}.pth")
try:
    plot(metrics_log)
except Exception as e:
    print(f"Grafik çizilirken hata oluştu: {e}")
    training_error_occurred=True

print(formatted_datetime)


