import os
import time
import numpy as np
from pick_place_env import PickPlaceCustomEnv

from rrt import RRT, RRTStar
from planner import Planner


## Enviroment Hyperparameters
random_object_pos = True # Randomize object position?
random_object_z_rot = False # Randomize object z-rotation?

## Initialize the environment
xml_path = os.path.join(os.path.dirname(__file__), "franka_emika_panda/scene_pickplace.xml")
env = PickPlaceCustomEnv(xml_path,
                         random_object_pos=random_object_pos,
                         random_object_z_rot=random_object_z_rot,
                         render_mode="human",
                         render_fps=30
                         )

## Initialize the planner
print(f"Joint Limits: {env.joint_limits}")
# motion_planner = RRT(env.joint_limits[:7], step_size=(0.05,), max_iter=2500)
motion_planner = RRTStar(env.joint_limits[:7], step_size=(0.05,), max_iter=2500, neighbor_radius=0.25)
planner = Planner(motion_planner, env.model, debug=False)

## Reset the environment
obs, info = env.reset()
print("Initial Observation:", obs)

step = 0
while True:
    start_time = time.time()

    print(f"Step: {step}")
    action = planner.plan(obs, env.data)  # Use the planner to get the action
    # action = None # Stay at home position
    # action = np.random.uniform(-1.0, 1.0, size=env.model.nu)  # Random action
    obs, done, info = env.step(action)
    # print(f"Joint Angles: {obs['state'][:9]}")
    # print(f"Grasp Site Pose: {obs['state'][9:12]} | Grasp Site Quat: {obs['state'][12:16]}")
    # print(f"Object Pose: {obs['object'][:3]} | Object Quat: {obs['object'][3:]}")
    # print(f"Goal Position: {obs['goal']}")
    # print("Info:", info)
    env.render()

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for step: {elapsed_time * 1000:.4f} ms")
    if elapsed_time < env.simulation_step_time:
        time.sleep(env.simulation_step_time - elapsed_time)

    print(f"=="*20)
    step += 1

    if done:
        print("Episode done!")
        time.sleep(2)
        obs, info = env.reset()
        planner.reset()
        step = 0
    
env.close()
