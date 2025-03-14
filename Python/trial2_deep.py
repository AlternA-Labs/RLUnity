import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# Hiperparametreler
MAX_STEPS = 500_000          # Toplam adım sayısı
BATCH_SIZE = 64              # Mini-batch boyutu
GAMMA = 0.99                 # İndirim faktörü
LEARNING_RATE = 1e-3         # Öğrenme hızı
TARGET_UPDATE_FREQ = 1000    # Target network güncelleme sıklığı (adım)
REPLAY_BUFFER_CAPACITY = 100_000

# Epsilon-greedy parametreleri
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.999995  # Her adımda çarpılacak

# Cihaz ayarı (GPU varsa kullanılır)
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# Replay buffer tanımı
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

# Basit DQN mimarisi (MLP)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)
print('unity')
# Unity ortamına bağlanıyoruz (Play modunda çalışmalı)
env = UnityEnvironment(file_name=None)
env.reset()
behavior_name = list(env.behavior_specs.keys())[0]
print('ekin')
spec = env.behavior_specs[behavior_name]

# Durum (observation) boyutu: İlk gözlemin shape'ını alıyoruz ve flatten ediyoruz
obs_shape = spec.observation_specs[0].shape
state_dim = int(np.prod(obs_shape))

# Discrete aksiyon için: Tek branch varsayıyoruz
num_actions = spec.action_spec.discrete_branches
print(f"State dim: {state_dim}, Num Actions: {num_actions}")

# Policy ve target network’leri oluşturuyoruz
policy_net = DQN(state_dim, num_actions).to(device)
target_net = DQN(state_dim, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

# Epsilon başlangıç değeri
epsilon = EPS_START

# Toplam adım sayacı
step_count = 0
episode_count = 0

def get_agent_state(decision_steps, agent_id):
    # İlk gözlem vektörünü flatten ediyoruz
    return decision_steps[agent_id].obs[0].flatten()

# Eğitim döngüsü (episode bazlı)
while step_count < MAX_STEPS:
    # Yeni episode başlat (Unity reset ile)
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    # Tek agent varsaydığımız için ilk agent id'sini alıyoruz
    agent_id = list(decision_steps.agent_id_to_index.keys())[0]  # agent_id_to_index dict'i ML-Agents 0.28+ sürümünde var
    state = get_agent_state(decision_steps, agent_id)
    episode_reward = 0
    done = False

    while not done and step_count < MAX_STEPS:
        # Epsilon-greedy aksiyon seçimi
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

        # Discrete aksiyon: shape (num_agents, num_branches) => (1, 1)
        action_array = np.array([[action]])
        action_tuple = ActionTuple(discrete=action_array)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        # Yeni adımın gözlemleri
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if agent_id in terminal_steps:
            next_state = terminal_steps[agent_id].obs[0].flatten()
            reward = terminal_steps[agent_id].reward
            done = True
        else:
            next_state = decision_steps[agent_id].obs[0].flatten()
            reward = decision_steps[agent_id].reward
            done = False

        # Geçişi replay buffer'a ekle
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        step_count += 1

        # Epsilon'u adım bazında azaltıyoruz
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Replay buffer yeterince dolu ise eğitim adımını gerçekleştir
        if len(replay_buffer) >= BATCH_SIZE:
            transitions = replay_buffer.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            # Tüm durumları tensor haline getiriyoruz
            state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
            action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
            reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
            next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
            done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(device)

            # Mevcut Q değerlerini hesaplıyoruz
            current_q = policy_net(state_batch).gather(1, action_batch)

            # Hedef Q değeri = reward + gamma * max_a' target_net(next_state)
            next_q = target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q = reward_batch + (1 - done_batch) * GAMMA * next_q

            loss = nn.MSELoss()(current_q, expected_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Target network güncellemesi (her TARGET_UPDATE_FREQ adımda)
        if step_count % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    episode_count += 1
    print(f"Episode: {episode_count}, Episode Reward: {episode_reward}, Total Steps: {step_count}, Epsilon: {epsilon:.3f}")

env.close()
print("Eğitim tamamlandı.")