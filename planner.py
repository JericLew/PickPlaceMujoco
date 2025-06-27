import time
import numpy as np
import inverse_kinematics as ik

class Planner():
    def __init__(self, motion_planner, mjModel, debug=False):
        self.state_list =["initial",
                          "moving_above_object",
                          "approaching_object",
                          "grasping_object",
                          "lifting_object",
                          "moving_above_goal",
                          "placing_object",
                          "releasing_object",
                          "moving_away_from_object",
                          "success"]
        
        # Variables
        self.debug = debug
        self.prev_task_plan_time = time.time()
        self.prev_motion_plan_time = time.time()
        self.prev_state_time = time.time()
        self.current_time = time.time()
        self.prev_state = "initial"
        self.state = "initial"
        self.grasped_pos = None # Position of the object when grasped
        self.planned_joint_actions = [] # List of planned joint actions
        self.planned_joint_actions_index = 0 # Index of the planned joint action to execute
        self.target_gripper_dist = 255.0 # Default gripper position (open)
        
        # Motion planner and MuJoCo model
        self.motion_planner = motion_planner
        self.mjModel = mjModel

        # Constants for distance checks
        self.task_replanning_interval = 2.0  # Time interval for task replanning
        self.motion_repanning_interval = 0.5  # Time interval for motion replanning motion
        self.state_timeout = 5.0  # Time to wait before replanning if no progress is made
        self.z_above_theta = 0.05 # Height offset from target height
        self.xy_above_threshold = 0.015 # Distance threshold for checking if a is near b in the xy plane
        self.z_above_threshold = 0.1  # Height threshold for checking if a is above b in the z direction
        self.near_threshold = 0.020  # Distance threshold for checking if a is near b
        self.joint_angle_threshold = 0.01  # Threshold for joint angle changes to consider a new action

    def _check_timeout(self):
        return self.current_time - self.prev_state_time > self.state_timeout

    def _check_a_above_b(self, a_pos, b_pos):
        return np.linalg.norm(a_pos[:2] - b_pos[:2]) < self.xy_above_threshold and \
               (a_pos[2] - b_pos[2]) > self.z_above_threshold
    
    def _check_a_near_b(self, a_pos, b_pos):
        return np.linalg.norm(a_pos - b_pos) < self.near_threshold

    def _check_grasped(self, gripper_dist):
        return gripper_dist < 0.0255 # NOTE magic num

    def _check_released(self, gripper_dist):
        return gripper_dist > 0.035 # NOTE magic num
    
    def plan(self, obs, mjData):
        ## Update the current time
        self.current_time = time.time()
        
        ## Extract information from the observation
        joint_angles = obs['state'][:7]  # 7 joint angles excluding the gripper
        gripper_dist = obs['state'][7] # Gripper distance (0.0 for closed, 0.04 for open)
        grasp_site_pos = obs['state'][9:12]  # Next 3 elements are grasp site position
        hand_quat = obs['state'][12:16]  # Next 4 elements are hand quaternion
        object_pos = obs['object'][:3]
        object_quat = obs['object'][3:]
        goal_pos = obs["goal"][:3]

        ## Task Planning
        # if self.debug:
        print(f"Time since last task plan: {self.current_time - self.prev_task_plan_time:.2f}s | Task interval: {self.task_replanning_interval}s")
        print(f"Time since last state change: {self.current_time - self.prev_state_time:.2f}s | State timeout: {self.state_timeout}s")

        ## Check if it's time to re-plan the task
        if self.state == "initial" or self.current_time - self.prev_task_plan_time > self.task_replanning_interval:
            self.prev_task_plan_time = self.current_time
            
            ## Task planning finite state machine
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
                    self.state = "moving_above_goal"
            
            elif self.state == "moving_above_goal":
                if not self._check_a_near_b(grasp_site_pos, object_pos):
                    self.state = "moving_above_object"
                elif self._check_a_above_b(grasp_site_pos, goal_pos):
                    self.state = "placing_object"
            
            elif self.state == "placing_object":
                if not self._check_a_near_b(grasp_site_pos, object_pos):
                    self.state = "moving_above_object"
                elif self._check_a_near_b(object_pos, goal_pos):
                    self.state = "releasing_object"
            
            elif self.state == "releasing_object":
                if not self._check_released(gripper_dist):
                    self.state = "success"
            
            elif self.state == "moving_away_from_object":
                if not self._check_a_near_b(grasp_site_pos, object_pos) and self._check_a_near_b(object_pos, goal_pos):
                    self.state = "success"

            elif self.state == "success":
                pass
            
            else:
                raise ValueError(f"Unknown state: {self.state}")
            
        
        ## Motion Planning    
        # if self.debug:
        print(f"Time since last motion plan: {self.current_time - self.prev_motion_plan_time:.2f}s | Motion interval: {self.motion_repanning_interval}s")

        ## Check if it's time to re-plan motion
        if self.state != self.prev_state or \
           self.current_time - self.prev_motion_plan_time > self.motion_repanning_interval:
            self.prev_motion_plan_time = self.current_time
            self.planned_joint_actions_index = 0

            ## Determine the target joint angles and gripper position based on the current state
            target_joint_angles = joint_angles.copy()  # Default target joint angles are the current ones
            self.target_gripper_dist = 255.0  # Default gripper position (open)
            target_grasp_site_pos = grasp_site_pos.copy()  # Default grasp site position is the current one
            target_hand_quat = np.array([-0.00484384,  0.707124,   0.7070547,  -0.00508084]) # Default hand quaternion (original hand orientation) NOTE: magic num

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
                self.target_gripper_dist = 0.0

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
                self.target_gripper_dist = 0.0

            elif self.state == "moving_above_goal":
                target_grasp_site_pos = goal_pos.copy()
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
                self.target_gripper_dist = 0.0

            elif self.state == "placing_object":
                target_grasp_site_pos = goal_pos.copy()
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
                self.target_gripper_dist = 0.0

            elif self.state == "releasing_object":
                pass

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

            elif self.state == "success":
                pass

            else:
                raise ValueError(f"Unknown state: {self.state}")

            ## Call motion planner if required
            if (target_joint_angles != joint_angles[:7]).all():
                self.planned_joint_actions = self.motion_planner.get_actions(joint_angles, target_joint_angles)
            else:
                self.planned_joint_actions = [joint_angles[:7].copy()]

            ## Debugging information
            if self.debug:
                print(f"Current Joint Angles: {joint_angles} | Current Gripper Position: {gripper_dist}")
                print(f"Target Joint Angles: {target_joint_angles} | Target Gripper Position: {self.target_gripper_dist}")
                print(f"Grasp Site Position: {grasp_site_pos} | Hand Quaternion: {hand_quat}")
                print(f"Target Grasp Site Position: {target_grasp_site_pos} | Target Hand Quaternion: {target_hand_quat}")
                print(f"Grasped Object Position: {self.grasped_pos} | Goal Position: {goal_pos}")
                print(f"Object Position: {object_pos} | Object Quaternion: {object_quat}")

        ## Prepare the action
        joint_action = joint_angles[:7].copy()  # Default action is to maintain current joint angles
        if len(self.planned_joint_actions) == 1: # RRT failed to find a path
            joint_action = self.planned_joint_actions[0]
        elif self.planned_joint_actions_index + 1 < len(self.planned_joint_actions): # Use the next planned joint action
            joint_action = self.planned_joint_actions[self.planned_joint_actions_index + 1]
            print(f"Ave. Joint Angle Diff: {np.mean(np.abs(joint_action - joint_angles[:7])):.4f}")
            if np.all(np.abs(joint_action - joint_angles[:7]) < self.joint_angle_threshold): # Only increment if close enough
                self.planned_joint_actions_index += 1
        else: # Use the last planned joint action
            joint_action = self.planned_joint_actions[-1]

        action = np.concatenate([joint_action, [self.target_gripper_dist]])

        if self.debug:
            print(f"Planned Joint Actions Index: {self.planned_joint_actions_index} | Length: {len(self.planned_joint_actions)}")
            print(f"Action: {action}")

        print(f"Previous State: {self.prev_state} | Current State: {self.state}")

        ## Update variables
        if self.state != self.prev_state:
            self.prev_state_time = self.current_time
            self.prev_state = self.state

        return action
