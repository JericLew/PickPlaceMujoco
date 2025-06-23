import numpy as np
import mujoco
import mujoco.viewer

class PickPlaceCustomEnv():
    def __init__(self, xml_path, random_object_pos=False, render_mode="human"):
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

        ## Option Flags
        self.random_object_pos = random_object_pos

        ## Define action and observation space
        # action_dim = self.model.nu # number of actuators/controls = dim(ctrl)
        # action_low = self.model.actuator_ctrlrange[:, 0].copy()
        # action_high = self.model.actuator_ctrlrange[:, 1].copy()

        ## Constants
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.object_body_id]]

        ## Data for rewards calculation
        self.initial_object_pos = None
        self.target_object_pos = None
        self.current_object_pos = None
        self.current_object_vel = None

        ## Termination flags
        self.success = False

        ## Rendering
        self.render_mode = render_mode
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
            object_delta_x_pos = 0.05
            object_delta_y_pos = 0.0
        new_object_pos = self.model.body("object").pos.copy()
        new_object_pos[:2] += (object_delta_x_pos, object_delta_y_pos)
        self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3] = new_object_pos

        mujoco.mj_forward(self.model, self.data)

        ## Set data for rewards calculation
        self.target_object_pos = self.model.body("target").pos.copy()
        self.initial_object_pos = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 3].copy()
        self.current_object_pos = self.initial_object_pos.copy()
        self.current_object_vel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 3].copy()

        ## Set termination flags
        self.success = False

        obs = self._get_obs()

        ## Prep info
        info = dict()
        info["success"] = self.success

        return obs, info

    def step(self, action):
        # self.data.ctrl[:] = action # takes in absolute joint angles

        ## Handle action
        # self.data.ctrl[:] = action
        
        ## Step simulation
        mujoco.mj_step(self.model, self.data, 5) # 5 substeps 2ms each = 10ms total

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
        ## Robot state information
        robot_joint_angles = self.data.qpos[:8].copy().astype(np.float32) # 8 joint angles
        hand_pos = self.data.xpos[self.hand_id].copy().astype(np.float32) # 3 hand position
        hand_quat = self.data.xquat[self.hand_id].copy().astype(np.float32) # 4 hand quaternion
        state = np.concatenate([robot_joint_angles, hand_pos, hand_quat]).astype(np.float32) # (8 + 3 + 4 = 15)

        ## Object information
        object = self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 7].copy().astype(np.float32) # object position and quaternion
        # object_qvel = self.data.qvel[self.object_qpos_addr : self.object_qpos_addr + 7] # object velocity
        
        ## Target information
        target_pos = self.target_object_pos.copy().astype(np.float32) # target position

        obs = {"state": state, "object": object, "target": target_pos}
        return obs
    
    def _get_done(self):
        done = self.success
        return done
    
    def _check_success(self):
        in_target = np.linalg.norm(self.current_object_pos[:2] - self.target_object_pos[:2]) < 0.10
        still = np.linalg.norm(self.current_object_vel) < 0.01 # NOTE arbitrary speed threshold
        return in_target # and still
        
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    import os

    ## Enviroment Hyperparameters
    random_object_pos = True # Randomize object position?
    max_episode_steps = 10000

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
