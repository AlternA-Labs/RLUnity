# ===== inference_only.py =====
import os
import time
import numpy as np
import torch
import torch.nn as nn
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

########################################
# Ayarlar – yalnızca bunları değiştir
########################################
MODEL_PATH ='models/actor2025-05-07_12:46.pth'
    #"models/actor2025-05-05_11:24.pth"
    #"models/actor2025-04-29_09:56.pth"   # <- bu şimdilik en iyisiydi.
UNITY_EXEC = None          # Editor’da Play modundaysan None bırak
BASE_PORT  = 5004          # ProjectSettings/ML-Agents → Editor Port
MAX_EPISODES = 20          # Kaç tam bölüm koşturalım?
RENDER_EVERY_STEP = True  # Unity tarafında otomatik render açıksa True’ya gerek yok
NOISE_STD = 0.0            # İstersen hafif exploration ekle (örn. 0.05)


device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
########################################
# Ağ tanımı (eğitimde kullandığınla birebir)
########################################
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

########################################
# Ortamı başlat
########################################
env_params = EnvironmentParametersChannel()
print("Unity başlatılıyor…")
env = UnityEnvironment(file_name=None, base_port=5004, side_channels=[env_params])
env_params.set_float_parameter("is_training", 0.0)#test
env.reset()

behavior_name = list(env.behavior_specs.keys())[0]
spec          = env.behavior_specs[behavior_name]
state_dim     = int(np.prod(spec.observation_specs[0].shape))
action_dim    = spec.action_spec.continuous_size

print(f"Behavior: {behavior_name} | state_dim: {state_dim} | action_dim: {action_dim}")

########################################
# Modeli yükle
########################################
actor = Actor(state_dim, action_dim).to(device)
actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
actor.eval()
print(f"Model yüklendi: {MODEL_PATH}")

########################################
# Yardımcı fonksiyonlar
########################################
def preprocess(obs_batch: np.ndarray) -> torch.Tensor:
    """
    Eğitimde yaptığın basit normalizasyonu yeniden uygula.
    obs_batch: (N, state_dim) numpy
    """
    obs_batch = obs_batch.copy()
    obs_batch[:, 0]  /= 10.0   # dir.magnitude
    obs_batch[:, 1]  /= 10.0   # fwdSpeed
    obs_batch[:, -5:] /= 10.0  # son 5 raycast
    return torch.FloatTensor(obs_batch).to(device)

def act(state_np: np.ndarray) -> np.ndarray:
    state_t = preprocess(state_np)
    with torch.no_grad():
        action = actor(state_t).cpu().numpy()
    if NOISE_STD > 0:
        action += np.random.normal(0, NOISE_STD, size=action.shape)
    return np.clip(action, -1.0, 1.0)

########################################
# Episode döngüsü
########################################
episode     = 0
global_step = 0
episode_return = 0.0
decision_steps, terminal_steps = env.get_steps(behavior_name)

# Agent’ların son hallerini sakla
agent_states  = {aid: ds.obs[0].flatten() for aid, ds in decision_steps.items()}
agent_returns = {aid: 0.0 for aid in agent_states}

while episode < MAX_EPISODES:
    # --- 1) Aksiyon üret ---
    agent_ids = list(decision_steps.agent_id_to_index.keys())
    states_np = np.vstack([agent_states[aid] for aid in agent_ids])  # (N, state_dim)
    actions_np = act(states_np)

    # sıra bozulmasın diye aynı dizide aksiyonları yerleştir
    action_tuple = ActionTuple(continuous=actions_np.astype(np.float32))
    env.set_actions(behavior_name, action_tuple)

    # --- 2) Ortamı ilerlet ---
    env.step()
    global_step += 1

    # --- 3) Yeni gözlemler ---
    next_decision_steps, next_terminal_steps = env.get_steps(behavior_name)

    # Önce terminal olanlar
    for aid in next_terminal_steps:
        reward = next_terminal_steps[aid].reward
        agent_returns[aid] += reward
        print(f"[Episode {episode}] Agent {aid} bitti, reward={agent_returns[aid]:.2f}")
        # episode sonu kontrolü: tüm agent'lar bittiyse yeni episode’a geç
    # Ardından devam edenler
    for aid in next_decision_steps:
        agent_states[aid]  = next_decision_steps[aid].obs[0].flatten()
        agent_returns[aid] += next_decision_steps[aid].reward

    # Episode tamamlandı mı?
    if len(next_terminal_steps) > 0 and all(aid in next_terminal_steps for aid in agent_states):
        mean_return = np.mean(list(agent_returns.values()))
        print(f"===> Episode {episode} bitti | ortalama ödül: {mean_return:.2f} | adım: {global_step}")
        episode += 1
        episode_return = 0.0
        # Agent sözlüklerini sıfırla (Unity reset’i otomatik)
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_states  = {aid: ds.obs[0].flatten() for aid, ds in decision_steps.items()}
        agent_returns = {aid: 0.0 for aid in agent_states}
    else:
        decision_steps = next_decision_steps  # devam eden ajanlar için güncelle

    if RENDER_EVERY_STEP:
        time.sleep(1/30)  # izlemeyi kolaylaştırmak için 30 FPS civarı

print("Kontrol döngüsü tamamlandı, Unity ortamı kapatılıyor…")
env.close()
