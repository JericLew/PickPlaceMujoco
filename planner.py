import time
import numpy as np
import inverse_kinematics as ik

class Planner():
    def __init__(self, motion_planner, mjModel):
        self.state_list =["initial",
                          "moving_above_object",
                          "approaching_object",
                          "grasping_object",
                          "lifting_object",
                          "moving_above_target",
                          "placing_object",
                          "releasing_object",
                          "moving_away_object",
                          "success"]
        
        # Variables
        self.prev_plan_time = time.time()
        self.prev_state_time = time.time()
        self.current_time = time.time()
        self.prev_state = "initial"
        self.state = "initial"
        self.grasped_pos = None
        
        # Motion planner and MuJoCo model
        self.motion_planner = motion_planner
        self.mjModel = mjModel

        # Constants for distance checks
        self.replanning_interval = 2.0  # Time interval for replanning
        self.state_timeout = 5.0  # Time to wait before replanning if no progress is made
        self.z_above_theta = 0.05 # Height above the object when moving above it
        self.xy_above_threshold = 0.020 # Distance threshold for checking if a is above b in the xy plane
        self.z_above_threshold = 0.1  # Height threshold for checking if a is above b in the z direction
        self.near_threshold = 0.020  # Distance threshold for checking if a is near b

    def _check_timeout(self):
        return time.time() - self.prev_state_time > self.state_timeout

    def _check_a_above_b(self, a_pos, b_pos):
        return np.linalg.norm(a_pos[:2] - b_pos[:2]) < self.xy_above_threshold and \
               (a_pos[2] - b_pos[2]) > self.z_above_threshold
    
    def _check_a_near_b(self, a_pos, b_pos):
        return np.linalg.norm(a_pos - b_pos) < self.near_threshold

    def _check_grasped(self, gripper_dist):
        return gripper_dist < 0.0255

    def _check_released(self, gripper_dist):
        return gripper_dist > 0.035
    
    def plan(self, obs, mjData):
        self.current_time = time.time()
        
        ## Extract information from the observation
        joint_angles = obs['state'][:7]  # 7 joint angles excluding the gripper
        gripper_dist = obs['state'][7] # Gripper distance (0.0 for closed, 0.04 for open)
        grasp_site_pos = obs['state'][9:12]  # Next 3 elements are grasp site position
        hand_quat = obs['state'][12:16]  # Next 4 elements are hand quaternion
        object_pos = obs['object'][:3]
        object_quat = obs['object'][3:]
        target_pos = obs["target"][:3]

        ## Check if it's time to re-plan task
        print(f"Time since last plan: {self.current_time - self.prev_plan_time:.2f}s | Replanning interval: {self.replanning_interval}s")
        print(f"Prev State: {self.prev_state} | Current State: {self.state}")
        print(f"Time since last state change: {self.current_time - self.prev_state_time:.2f}s | State timeout: {self.state_timeout}s")
        if self.state == "initial" or self.current_time - self.prev_plan_time > self.replanning_interval:
            self.prev_plan_time = self.current_time
            
            ## Task planning logic
            ## This is a finite state machine that transitions between states
            if self.state == "initial":
                if self._check_a_above_b(grasp_site_pos, object_pos):
                    self.state = "approaching_object"
                else:
                    self.state = "moving_above_object"
            
            elif self.state == "moving_above_object":
                if self._check_a_above_b(grasp_site_pos, object_pos):
                    self.state = "approaching_object"
            
            elif self.state == "approaching_object":
                if self._check_a_near_b(grasp_site_pos, object_pos):
                    self.state = "grasping_object"
                elif self._check_timeout():
                    self.state = "moving_above_object"

            elif self.state == "grasping_object":
                if self._check_grasped(gripper_dist) and self._check_a_near_b(grasp_site_pos, object_pos):
                    self.grasped_pos = object_pos.copy()
                    self.state = "lifting_object"
                elif self._check_timeout():
                    self.state = "moving_above_object"
            
            elif self.state == "lifting_object":
                if not self._check_a_near_b(grasp_site_pos, object_pos):
                    self.state = "moving_above_object"
                elif self._check_a_above_b(object_pos, self.grasped_pos):
                    self.state = "moving_above_target"
            
            elif self.state == "moving_above_target":
                if not self._check_a_near_b(grasp_site_pos, object_pos):
                    self.state = "moving_above_object"
                elif self._check_a_above_b(grasp_site_pos, target_pos):
                    self.state = "placing_object"
            
            elif self.state == "placing_object":
                if not self._check_a_near_b(grasp_site_pos, object_pos):
                    self.state = "moving_above_object"
                elif self._check_a_near_b(object_pos, target_pos):
                    self.state = "releasing_object"
            
            elif self.state == "releasing_object":
                if not self._check_released(gripper_dist):
                    self.state = "success"
            
            elif self.state == "moving_away_object":
                if not self._check_a_near_b(grasp_site_pos, object_pos) and self._check_a_near_b(object_pos, target_pos):
                    self.state = "success"

            elif self.state == "success":
                pass
            
            else:
                raise ValueError(f"Unknown state: {self.state}")
            
            ## Update variables
            if self.state != self.prev_state:
                self.prev_state_time = time.time()
            self.prev_state = self.state
    
        
        ## Get the next action from the motion planner based on the current state
        target_joint_angles = joint_angles[:7].copy()  # Default action is to maintain current joint angles
        target_gripper_pos = 255.0  # Default gripper position (open)
        target_hand_quat = np.array([-0.00484384,  0.707124,   0.7070547,  -0.00508084])
        if self.state == "initial":
            pass
        elif self.state == "moving_above_object":
            target_grasp_site_pos = object_pos.copy()
            target_grasp_site_pos[2] += self.z_above_threshold + self.z_above_theta
            # target_hand_quat = hand_quat.copy()
            ikresults = ik.qpos_from_site_pose(
                mjmodel=self.mjModel, 
                mjdata=mjData, 
                site_name="grasp_site", 
                target_pos=target_grasp_site_pos, 
                target_quat=target_hand_quat, 
                joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            )
            target_joint_angles = ikresults[0][:7]
        elif self.state == "approaching_object":
            target_grasp_site_pos = object_pos.copy()
            # target_hand_quat = hand_quat.copy()
            ikresults = ik.qpos_from_site_pose(
                mjmodel=self.mjModel, 
                mjdata=mjData, 
                site_name="grasp_site", 
                target_pos=target_grasp_site_pos, 
                target_quat=target_hand_quat, 
                joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            )
            target_joint_angles = ikresults[0][:7]
        elif self.state == "grasping_object":
            target_gripper_pos = 0.0
        elif self.state == "lifting_object":
            target_grasp_site_pos = self.grasped_pos.copy()
            target_grasp_site_pos[2] += self.z_above_threshold + self.z_above_theta
            # target_hand_quat = hand_quat.copy()
            ikresults = ik.qpos_from_site_pose(
                mjmodel=self.mjModel, 
                mjdata=mjData, 
                site_name="grasp_site", 
                target_pos=target_grasp_site_pos, 
                target_quat=target_hand_quat, 
                joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            )
            target_joint_angles = ikresults[0][:7]
            target_gripper_pos = 0.0
        elif self.state == "moving_above_target":
            target_grasp_site_pos = target_pos.copy()
            target_grasp_site_pos[2] += self.z_above_threshold + self.z_above_theta
            # target_hand_quat = hand_quat.copy()
            ikresults = ik.qpos_from_site_pose(
                mjmodel=self.mjModel, 
                mjdata=mjData, 
                site_name="grasp_site", 
                target_pos=target_grasp_site_pos, 
                target_quat=target_hand_quat, 
                joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            )
            target_joint_angles = ikresults[0][:7]
            target_gripper_pos = 0.0
        elif self.state == "placing_object":
            target_grasp_site_pos = target_pos.copy()
            # target_hand_quat = hand_quat.copy()
            ikresults = ik.qpos_from_site_pose(
                mjmodel=self.mjModel, 
                mjdata=mjData, 
                site_name="grasp_site", 
                target_pos=target_grasp_site_pos, 
                target_quat=target_hand_quat, 
                joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            )
            target_joint_angles = ikresults[0][:7]
            target_gripper_pos = 0
        elif self.state == "releasing_object":
            target_gripper_pos = 255.0
        elif self.state == "moving_away_object":
            target_grasp_site_pos = object_pos.copy()
            target_grasp_site_pos[2] += self.z_above_threshold + self.z_above_theta
            # target_hand_quat = hand_quat.copy()
            ikresults = ik.qpos_from_site_pose(
                mjmodel=self.mjModel, 
                mjdata=mjData, 
                site_name="grasp_site", 
                target_pos=target_grasp_site_pos, 
                target_quat=target_hand_quat, 
                joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            )
            target_joint_angles = ikresults[0][:7]
            target_gripper_pos = 255.0
        elif self.state == "success":
            pass

        print(f"Planner State: {self.state}")
        print(f"Current Joint Angles: {joint_angles} | Target Joint Angles: {target_joint_angles} | Gripper Position: {target_gripper_pos}")
        print(f"Current Grasp Site Position: {grasp_site_pos} | Current Object Position: {object_pos}")
        print(f"Grasped Object Position: {self.grasped_pos} | Target Position: {target_pos}")
        print(f"Current Hand Quaternion: {hand_quat} | Target Hand Quaternion: {target_hand_quat}")

        joint_action = self.motion_planner.get_action(joint_angles, target_joint_angles)
        action = np.concatenate([joint_action, [target_gripper_pos]])  # Append gripper action

        print(f"Action: {action}")
        return action
