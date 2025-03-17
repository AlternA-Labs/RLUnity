import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import datetime
import pytz

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

##########################################
# Hiperparametreler
##########################################
MAX_STEPS = 200_000             # Toplam adım sayısı
BATCH_SIZE = 128                # Mini-batch boyutu
GAMMA = 0.99                    # İndirim faktörü
LEARNING_RATE = 5e-3            # Öğrenme hızı
REPLAY_BUFFER_CAPACITY = 100_000
TAU = 0.005                     # Soft update katsayısı
NOISE_STD = 0.15                # Gaussian gürültü için standart sapma

device = torch.device("mps")  # "mps" ya da "cuda" da kullanılabilir

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

##########################################
# Prioritized Replay Buffer
##########################################
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.alpha = alpha  # 0 = uniform, 1 = full prioritization
        self.pos = 0

    def push(self, *args):
        max_priority = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
            self.priorities.append(max_priority)
        else:
            self.memory[self.pos] = Transition(*args)
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array([float(p) for p in self.priorities], dtype=np.float64)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-5

##########################################
# Actor ve Critic Ağları
##########################################
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Çıktı [-1, 1] aralığında
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

##########################################
# Unity Ortamını Başlatma
##########################################
print('Waiting for Unity environment...')
env = UnityEnvironment(file_name=None, base_port=5004)
env.reset()
print('Unity environment ready!')

# Behavior bilgilerini al
behavior_names = list(env.behavior_specs.keys())
behavior_name = behavior_names[0]
spec = env.behavior_specs[behavior_name]

# Observation ve action boyutlarını çıkar
obs_shape = spec.observation_specs[0].shape
state_dim = int(np.prod(obs_shape))
action_dim = spec.action_spec.continuous_size

print(f"Behavior: {behavior_name}")
print(f"State dim: {state_dim}, Action dim: {action_dim}")

##########################################
# Ağlar, Optimizer ve Replay Buffer
##########################################
replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_CAPACITY)

actor = Actor(state_dim, action_dim).to(device)
critic = Critic(state_dim, action_dim).to(device)
actor_target = Actor(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)

actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

##########################################
# Policy Distillation için Best Actor
##########################################
best_actor = copy.deepcopy(actor)
best_reward = -float('inf')  # İlk başta en iyi ödül -sonsuz

##########################################
# Yardımcı Fonksiyon: Soft Update
##########################################
def soft_update(target_net, source_net, tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

##########################################
# Ajan Durumlarını Takip
##########################################
agent_states = {}     # agent_id -> son state
agent_rewards = {}    # agent_id -> birikmiş ödül
agent_done_flags = {} # agent_id -> done flag

decision_steps, terminal_steps = env.get_steps(behavior_name)
for agent_id in decision_steps:
    obs = decision_steps[agent_id].obs[0].flatten()
    agent_states[agent_id] = obs
    agent_rewards[agent_id] = 0.0
    agent_done_flags[agent_id] = False

global_step = 0
episode_rewards = []  # Geçmiş episod ödüllerini izlemek için

##########################################
# Eğitim Döngüsü
##########################################
while global_step < MAX_STEPS:
    # 1) Aksiyon oluşturma
    agent_ids = list(decision_steps.agent_id_to_index.keys())
    actions_dict = {}
    for agent_id in agent_ids:
        state = agent_states[agent_id]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            base_action = actor(state_tensor).cpu().numpy()[0]
        noise = np.random.normal(0, NOISE_STD, size=action_dim)
        action = base_action + noise
        action = np.clip(action, -1.0, 1.0)
        actions_dict[agent_id] = action

    # 2) Aksiyonları environment'a gönderme
    action_array = np.zeros((len(agent_ids), action_dim), dtype=np.float32)
    for i, agent_id in enumerate(agent_ids):
        action_array[i, :] = actions_dict[agent_id]
    action_tuple = ActionTuple(continuous=action_array)
    env.set_actions(behavior_name, action_tuple)

    # 3) Ortamı bir adım ilerlet
    try:
            env.step()

    except Exception as e:
        print(f"Error during environment step: {e}")
        training_error_occurred = True
        break
    global_step += 1

    # 4) Yeni step sonuçları al
    next_decision_steps, next_terminal_steps = env.get_steps(behavior_name)

    # 4a) Terminal olan agentlar
    for agent_id in next_terminal_steps:
        reward = next_terminal_steps[agent_id].reward
        next_obs = next_terminal_steps[agent_id].obs[0].flatten()
        done = True

        old_state = agent_states[agent_id]
        old_action = actions_dict[agent_id]
        replay_buffer.push(old_state, old_action, reward, next_obs, done)

        agent_rewards[agent_id] += reward
        print(f"[Step {global_step}] Agent {agent_id} done. Episode Reward = {agent_rewards[agent_id]:.2f}")
        episode_rewards.append(agent_rewards[agent_id])

        agent_states[agent_id] = next_obs
        agent_rewards[agent_id] = 0.0
        agent_done_flags[agent_id] = True

    # 4b) Kararda (decision) olan agentlar
    for agent_id in next_decision_steps:
        reward = next_decision_steps[agent_id].reward
        next_obs = next_decision_steps[agent_id].obs[0].flatten()
        done = False

        if agent_id in actions_dict:
            old_state = agent_states[agent_id]
            old_action = actions_dict[agent_id]
            replay_buffer.push(old_state, old_action, reward, next_obs, done)

        agent_states[agent_id] = next_obs
        agent_rewards[agent_id] += reward
        agent_done_flags[agent_id] = False

    decision_steps, terminal_steps = next_decision_steps, next_terminal_steps

    # 5) Eğitim: Replay Buffer yeterli ise güncelle
    if len(replay_buffer.memory) >= BATCH_SIZE:
        transitions, indices, weights = replay_buffer.sample(BATCH_SIZE, beta=0.4)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.FloatTensor(np.array(batch.action)).to(device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(device)
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(device)

        # Critic güncellemesi
        with torch.no_grad():
            next_actions = actor_target(next_state_batch)
            target_q = critic_target(next_state_batch, next_actions)
            expected_q = reward_batch + (1 - done_batch) * GAMMA * target_q

        current_q = critic(state_batch, action_batch)
        td_errors = (current_q - expected_q).detach().cpu().numpy()
        critic_loss = ((current_q - expected_q) ** 2 * weights_tensor).mean()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Prioriteleri güncelle
        replay_buffer.update_priorities(indices, td_errors)

        # Actor güncellemesi (Distillation eklenmiş)
        actor_loss = -critic(state_batch, actor(state_batch)).mean()
        # Distillation loss: actor çıktısı best_actor çıktısına yakın olsun
        distill_loss = nn.MSELoss()(actor(state_batch), best_actor(state_batch))
        total_actor_loss = actor_loss + 0.01 * distill_loss

        actor_optimizer.zero_grad()
        total_actor_loss.backward()
        actor_optimizer.step()

        # Target ağların soft update'u
        soft_update(actor_target, actor, TAU)
        soft_update(critic_target, critic, TAU)

    # 6) Best Actor güncellemesi (her 1000 adımda)
    if global_step % 1000 == 0 and episode_rewards:
        avg_reward = np.mean(episode_rewards[-10:])  # son 10 episodun ortalaması
        print(f"Global Step: {global_step}, Recent Avg Reward: {avg_reward:.2f}")
        print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_actor.load_state_dict(actor.state_dict())
            print(f"Best actor updated at step {global_step} with reward {best_reward:.2f}")

##########################################
# Eğitim Sonu: Model Kaydetme
##########################################
env.close()
print("Training completed.")

utc_plus_3 = pytz.timezone('Europe/Istanbul')
date = datetime.datetime.now(utc_plus_3)
formatted_datetime = date.strftime("%Y-%m-%d_%H:%M")

if not training_error_occurred:
    os.makedirs("models", exist_ok=True)
    torch.save(actor.state_dict(), f"models/actor_{formatted_datetime}.pth")
    torch.save(critic.state_dict(), f"models/critic_{formatted_datetime}.pth")
    print(f"Models saved: actor_{formatted_datetime}.pth, critic_{formatted_datetime}.pth")
else:
    print("Training error occurred. Models are not saved.")