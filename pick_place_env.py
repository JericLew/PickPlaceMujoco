import time
import numpy as np

import mujoco
import mujoco.viewer

class PickPlaceCustomEnv():
    def __init__(self, xml_path, random_object_pos=False, simulation_steps=10, render_mode="human", render_fps=30):
        print("Initializing PickPlaceCustomEnv...")
        self.np_random = None

        self.colors = ['red', 'green', 'blue']
        self.color_map = {
            'red': [1.0, 0.0, 0.0, 1.0],
            'green': [0.0, 1.0, 0.0, 1.0],
            'blue': [0.0, 0.0, 1.0, 1.0]
        }

        ## Load MuJoCo model and data
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        ## Simulation parameters
        self.simulation_steps = simulation_steps  # Number of substeps for each step
        self.simulation_step_time = 0.002 * simulation_steps  # Each mujoco step is 2ms, so total time for one env step is 2ms * simulation_steps

        ## Option Flags
        self.random_object_pos = random_object_pos

        ## Define action and observation space
        action_dim = self.model.nu # number of actuators/controls = dim(ctrl)
        action_low = self.model.actuator_ctrlrange[:, 0].copy()
        action_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.joint_limits = np.array([action_low, action_high]).T  # shape (nu, 2)

        ## Constants
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.grasp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.object_body_id]]
        self.object_geom_height = self.model.geom("object_geom").size[2]  # Height of the object geom
        self.goal_geom_radius = self.model.geom("r_goal_geom").size[0]  # Radius of the goal geom
        self.goal_geom_height = self.model.geom("r_goal_geom").size[1]  # Height of the goal geom

        ## Data for status checking
        self.goal_pos = None
        self.current_object_pos = None
        self.current_object_vel = None

        ## Termination flags
        self.success = False

        ## Rendering
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.last_render_time = 0.0
        self.viewer = None

    def reset(self, seed=None, options=None):
        self.np_random = np.random.default_rng(seed)

        ## Options
        if options is not None:
            self.random_object_pos = options.get("random_object_pos", self.random_object_pos)

        ## Reset MuJoCo model and data to home position
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.keyframe('home').qpos
        self.data.ctrl[:] = self.model.keyframe('home').ctrl

        ## Randomly select object color
        object_color_name = self.np_random.choice(self.colors)
        object_color = self.color_map[object_color_name]
        self.model.geom("object_geom").rgba = object_color

        ## Set object position
        if self.random_object_pos: # Random object position
            object_delta_x_pos = self.np_random.uniform(-0.15, 0.15)
            object_delta_y_pos = self.np_random.uniform(-0.15, 0.15)
        else: # Fixed object position
            object_delta_x_pos = 0.0
            object_delta_y_pos = 0.0
        new_object_pos = self.model.body("object").pos.copy()
        new_object_pos[:2] += (object_delta_x_pos, object_delta_y_pos)
        self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3] = new_object_pos

        mujoco.mj_forward(self.model, self.data)

        ## Data for status checking
        self.goal_pos = self.model.body(f"{object_color_name[0]}_goal").pos.copy()\
            + np.array([0.0, 0.0, self.goal_geom_height + self.object_geom_height]) # Goal is at top of block, so add heights offset
        self.current_object_pos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3].copy()
        self.current_object_vel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 3].copy()

        ## Set termination flags
        self.success = False

        obs = self._get_obs()

        ## Prep info
        info = dict()
        info["success"] = self.success

        return obs, info

    def step(self, action):
        ## Handle action
        self.data.ctrl[:] = action
        
        ## Step simulation
        mujoco.mj_step(self.model, self.data, self.simulation_steps) # 2ms each 

        ## Update object info
        self.current_object_pos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # x,y,z
        self.current_object_vel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 3].copy() # dx,dy,dz

        ## Check termination conditions
        self.success = self._check_success()

        obs = self._get_obs()
        done = self._get_done()

        ## Prep info
        info = dict()
        info["success"] = self.success

        return obs, done, info
    
    def _get_obs(self):
        """
        "state": (16,) robot state information (9 joint angles, 3 grasp site position, 4 hand quaternion)
        "object": (7,) object position (x,y,z) and quaternion (qx,qy,qz,qw)
        "goal": (3,) goal position (x,y,z)
        """
        ## Robot state information
        robot_joint_angles = self.data.qpos[:9].copy().astype(np.float32) # 9 joint angles (7 for the arm, 2 for the gripper)
        grasp_site_pos = self.data.site_xpos[self.grasp_site_id].copy().astype(np.float32) # 3 grasp site position
        hand_quat = self.data.xquat[self.hand_id].copy().astype(np.float32) # 4 hand quaternion
        state = np.concatenate([robot_joint_angles, grasp_site_pos, hand_quat]).astype(np.float32) # (9 + 3 + 4) = 16

        ## Object information
        object = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 7].copy().astype(np.float32) # object position and quaternion
        # object_qvel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 7] # object velocity
        
        ## Goal information
        goal_pos = self.goal_pos.copy().astype(np.float32)

        obs = {"state": state, "object": object, "goal": goal_pos}
        return obs
    
    def _get_done(self):
        done = self.success
        return done
    
    def _check_success(self):
        near_goal_xy = np.linalg.norm(self.current_object_pos[:2] - self.goal_pos[:2]) < self.goal_geom_radius
        near_goal_z = np.abs(self.current_object_pos[2] - self.goal_pos[2]) < self.goal_geom_height + 2 * self.object_geom_height
        still = np.linalg.norm(self.current_object_vel) < 0.01 # NOTE arbitrary speed threshold
        gripper_is_far_from_object = np.linalg.norm(self.data.site_xpos[self.grasp_site_id] - self.current_object_pos) > 2 * self.object_geom_height
        return near_goal_xy and near_goal_z and still and gripper_is_far_from_object
        
    def render(self):
        time_now = time.time()
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if time_now - self.last_render_time >= 1.0 / self.render_fps:
                self.viewer.sync()
                self.last_render_time = time_now
            # self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    import os
    import time
    import numpy as np
    from pick_place_env import PickPlaceCustomEnv


    ## Enviroment Hyperparameters
    random_object_pos = True # Randomize object position?

    ## Initialize the environment
    xml_path = os.path.join(os.path.dirname(__file__), "franka_emika_panda/scene_pickplace.xml")
    env = PickPlaceCustomEnv(xml_path, 
                             random_object_pos=random_object_pos,
                             render_mode="human",
                             render_fps=3
                             )

    ## Reset the environment
    obs, info = env.reset()
    print("Initial Observation:", obs)

    step = 0
    while True:
        start_time = time.time()

        print(f"Step: {step}")
        action = np.random.uniform(-1.0, 1.0, size=env.model.nu)  # Random action
        obs, done, info = env.step(action)
        print(f"Joint Angles: {obs['state'][:9]}")
        print(f"Hand Pose: {obs['state'][9:12]} | Hand Quat: {obs['state'][12:16]}")
        print(f"Object Pose: {obs['object'][:3]} | Object Quat: {obs['object'][3:]}")
        print(f"Goal Position: {obs['goal']}")
        print("Info:", info)
        env.render()
        if done:
            print("Episode done!")
            break
        print(f"=="*20)
        step += 1

        elapsed_time = time.time() - start_time
        print(f"Elapsed time for step: {elapsed_time * 1000:.4f} ms")
        if elapsed_time < env.simulation_step_time:
            time.sleep(env.simulation_step_time - elapsed_time)

    env.close()
