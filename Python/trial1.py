import sys

from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
import numpy as np
print('unity')
# Unity Editor Play modundaki ortam ile bağlantı kurmak için file_name=None kullanıyoruz
env = UnityEnvironment(file_name=None,base_port=5004)
print('ekin')
print('sonuc;',list(env.behavior_specs.keys()))

# Bağlantıdaki tüm davranış (agent) isimlerini alıyoruz
env.reset()
behavior_names = list(env.behavior_specs.keys())
print("Bağlantıdaki davranışlar:", behavior_names)
# Örneğin çıktı: Bağlantıdaki davranışlar: ['RoketAgent']

# Burada ilk davranışı kullanıyoruz
behavior_name = behavior_names[0]
spec = env.behavior_specs[behavior_name]

# Bir episode (bölüm) boyunca ortamı çalıştıracağız
print("\nEpisode başlatıldı...")
decision_steps, terminal_steps = env.get_steps(behavior_name)

# Episode süresince, agent'ların karar vermesi gereken adımları işliyoruz
while len(terminal_steps) == 0:
    # Her bir agent'ın gözlemlerini yazdırıyoruz
    for agent_id, step in decision_steps.items():
        print(f"\nAgent {agent_id} gözlemleri:")
        # Gözlemler genellikle numpy dizileri şeklinde gelir, örneğin: [x konumu, y konumu, z konumu, ...]
        print(step.obs)  # Gözlem verilerini gösterir

    # Burada örnek olarak rastgele aksiyonlar üretiyoruz
    # spec.action_spec.continuous_size, ajan için beklenen aksiyon vektörünün boyutunu belirtir
    actions = np.random.randn(len(decision_steps), spec.action_spec.continuous_size)
    action = ActionTuple(continuous=actions)
    print("\nGönderilen aksiyonlar:")
    print(actions)

    # Üretilen aksiyonları Unity ortamına gönderiyoruz
    env.set_actions(behavior_name, action)
    env.step()  # Ortamı bir adım ileri taşıyoruz

    # Bir sonraki adımın gözlemleri ve sonlanmış (terminal) agent bilgilerini alıyoruz
    decision_steps, terminal_steps = env.get_steps(behavior_name)

# Episode sona erdikten sonra terminal adımları işliyoruz
print("\nEpisode sona erdi. Terminal adım verileri:")
for agent_id, terminal in terminal_steps.items():
    print(f"\nAgent {agent_id} terminal gözlemleri:")
    print(terminal.obs)
    print(f"Agent {agent_id} final ödül: {terminal.reward}")

# Ortamı kapatıyoruz
env.close()