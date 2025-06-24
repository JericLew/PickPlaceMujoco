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
                          "success"]
        
        # Variables
        self.state = "initial"
        self.grasped_pos = None
        
        # Motion planner and MuJoCo model
        self.motion_planner = motion_planner
        self.mjModel = mjModel

        # Constants for distance checks
        self.z_above_theta = 0.05 # Height above the object when moving above it
        self.xy_above_threshold = 0.005  # Distance threshold for checking if a is above b in the xy plane
        self.z_above_threshold = 0.1  # Height threshold for checking if a is above b in the z direction
        self.near_threshold = 0.001  # Distance threshold for checking if a is near b
        self.force_threshold = 0.1  # Force threshold for checking if the hand is grasping the object

    def _check_a_above_b(self, a_pos, b_pos):
        return np.linalg.norm(a_pos[:2] - b_pos[:2]) < self.xy_above_threshold and \
               (a_pos[2] - b_pos[2]) > self.z_above_threshold
    
    def _check_a_near_b(self, a_pos, b_pos):
        return np.linalg.norm(a_pos - b_pos) < self.near_threshold

    def _check_grasped(self, hand_pos, object_pos, hand_actuator_force):
        return hand_actuator_force > self.force_threshold and self._check_a_near_b(hand_pos, object_pos)
    
    def plan(self, obs, mjData):
        ## Extract information from the observation
        joint_angles = obs['state'][:8]
        actuator_forces = obs['state'][8:16]
        hand_actuator_force = actuator_forces[-1]  # Last actuator is the hand
        hand_pos = obs['state'][16:19]
        hand_quat = obs['state'][19:23]
        object_pos = obs['object'][:3]
        object_quat = obs['object'][3:]
        target_pos = obs["target"][:3]

        ## Task planning logic
        ## This is a finite state machine that transitions between states based on the hand and object positions
        if self.state == "initial":
            if self._check_a_above_b(hand_pos, object_pos):
                self.state = "approaching_object"
            else:
                self.state = "moving_above_object"
        elif self.state == "moving_above_object":
            if self._check_a_above_b(hand_pos, object_pos):
                self.state = "approaching_object"
        elif self.state == "approaching_object":
            if self._check_a_near_b(hand_pos, object_pos):
                self.state = "grasping_object"
        elif self.state == "grasping_object":
            if self._check_grasped(hand_pos, object_pos, hand_actuator_force):
                self.grasped_pos = object_pos.copy()
                self.state = "lifting_object"
        elif self.state == "lifting_object":
            if not self._check_grasped(hand_pos, object_pos, hand_actuator_force):
                self.state = "moving_above_object"
            elif self._check_a_above_b(object_pos, self.grasped_pos):
                self.state = "moving_above_target"
        elif self.state == "moving_above_target":
            if not self._check_grasped(hand_pos, object_pos, hand_actuator_force):
                self.state = "moving_above_object"
            elif self._check_a_above_b(hand_pos, target_pos):
                self.state = "placing_object"
        elif self.state == "placing_object":
            if not self._check_grasped(hand_pos, object_pos, hand_actuator_force):
                self.state = "moving_above_object"
            elif self._check_a_near_b(object_pos, target_pos):
                self.state = "releasing_object"
        elif self.state == "releasing_object":
            if not self._check_grasped(hand_pos, object_pos, hand_actuator_force):
                self.state = "success"
        elif self.state == "success":
            pass
        else:
            raise ValueError(f"Unknown state: {self.state}")

        ## Get the next action from the motion planner based on the current state
        target_joint_angles = joint_angles.copy()  # Default action is to maintain current joint angles
        if self.state == "initial":
            pass
        elif self.state == "moving_above_object":
            target_hand_pos = object_pos.copy()
            target_hand_pos[2] += self.z_above_threshold + self.z_above_theta
            target_hand_quat = hand_quat.copy()
            ikresults = ik.qpos_from_site_pose(
                mjmodel=self.mjModel, 
                mjdata=mjData, 
                site_name="hand", 
                target_pos=target_hand_pos, 
                target_quat=target_hand_quat, 
            )
            target_joint_angles = ikresults[0][:8]
        elif self.state == "approaching_object":
            target_hand_pos = object_pos.copy()
            target_hand_quat = hand_quat.copy()
            ikresults = ik.qpos_from_site_pose(
                mjmodel=self.mjModel, 
                mjdata=mjData, 
                site_name="hand", 
                target_pos=target_hand_pos, 
                target_quat=target_hand_quat, 
            )
            target_joint_angles = ikresults[0][:8]
        elif self.state == "grasping_object":
            target_joint_angles[-1] = 0.0  # Close the hand
        elif self.state == "lifting_object":
            target_hand_pos = self.grasped_pos.copy()
            target_hand_pos[2] += self.z_above_threshold + self.z_above_theta
            target_hand_quat = hand_quat.copy()
            ikresults = ik.qpos_from_site_pose(
                mjmodel=self.mjModel, 
                mjdata=mjData, 
                site_name="hand", 
                target_pos=target_hand_pos, 
                target_quat=target_hand_quat, 
            )
            target_joint_angles = ikresults[0][:8]
        elif self.state == "moving_above_target":
            target_hand_pos = target_pos.copy()
            target_hand_pos[2] += self.z_above_threshold + self.z_above_theta
            target_hand_quat = hand_quat.copy()
            ikresults = ik.qpos_from_site_pose(
                mjmodel=self.mjModel, 
                mjdata=mjData, 
                site_name="hand", 
                target_pos=target_hand_pos, 
                target_quat=target_hand_quat, 
            )
            target_joint_angles = ikresults[0][:8]
        elif self.state == "placing_object":
            target_hand_pos = target_pos.copy()
            target_hand_quat = hand_quat.copy()
            ikresults = ik.qpos_from_site_pose(
                mjmodel=self.mjModel, 
                mjdata=mjData, 
                site_name="hand", 
                target_pos=target_hand_pos, 
                target_quat=target_hand_quat, 
            )
            target_joint_angles = ikresults[0][:8]
        elif self.state == "releasing_object":
            target_joint_angles[-1] = 255.0  # Open the hand
        elif self.state == "success":
            pass
        action = self.motion_planner.get_action(joint_angles, target_joint_angles)
        
        print(f"Planner State: {self.state}")
        print(f"Current Joint Angles: {joint_angles}")
        print(f"Current Hand Position: {hand_pos}")
        print(f"Current Object Position: {object_pos}")
        print(f"Target Joint Angles: {target_joint_angles}")
        print(f"Action: {action}")
        return action
