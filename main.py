import os
import numpy as np
from pick_place_env import PickPlaceCustomEnv

## Enviroment Hyperparameters
random_object_pos = True # Randomize object position?
max_episode_steps = 10000000000000000000

xml_path = os.path.join(os.path.dirname(__file__), "franka_emika_panda/scene_pickplace.xml")
env = PickPlaceCustomEnv(xml_path, random_object_pos=random_object_pos, render_mode="human")
obs, info = env.reset()
print("Initial Observation:", obs)

for step in range(max_episode_steps):
    print(f"Step: {step}")
    action = np.random.uniform(-1.0, 1.0, size=env.model.nu)  # Random action
    obs, done, info = env.step(action)
    print(f"Joint Angles: {obs['state'][:8]}")
    print(f"Hand Pose: {obs['state'][8:11]} | Hand Quat: {obs['state'][11:15]}")
    print(f"Object Pose: {obs['object'][:3]} | Object Quat: {obs['object'][3:]}")
    print("Info:", info)
    env.render()
    if done:
        print("Episode done!")
        break
    print(f"=="*20)

env.close()