import numpy as np

class Planner():
    def __init__(self, motion_planner):
        self.state_list =["initial",
                          "move_above_object",
                          "approaching_object",
                          "grasping_object",
                          "lifting_object",
                          "moving_to_target",
                          "placing_object",
                          "release_object",
                          "success"]
        self.state = "initial"
        self.motion_planner = motion_planner
        self.xy_above_threshold = 0.025  # Distance threshold for checking if hand is above object
        self.z_above_threshold = 0.05  # Height threshold for checking if hand is above object
        self.grasp_threshold = 0.05  # Distance threshold for grasping
        self.grasped_pos = None  # Position of the object when grasped
        self.place_threshold = 0.05  # Distance threshold for placing the object

    def _check_above_object(self, hand_pos, object_pos):
        return np.linalg.norm(hand_pos[:2] - object_pos[:2]) < self.xy_above_threshold and \
               (hand_pos[2] - object_pos[2]) > self.z_above_threshold
    
    def _check_graspable(self, hand_pos, target_pos):
        return np.linalg.norm(hand_pos - target_pos) < self.grasp_threshold
    
    def _check_grasped(self, hand_pos, object_pos):
        # TODO check force sensing
        return np.linalg.norm(hand_pos - object_pos) < self.grasp_threshold
    
    def _check_lifted(self, object_pos):
        if self.grasped_pos is None:
            return False
        return np.linalg.norm(object_pos[:2] - self.grasped_pos[:2]) < self.xy_above_threshold and \
               (object_pos[2] - self.grasped_pos[2]) > self.z_above_threshold
    
    def _check_above_target(self, hand_pos, target_pos):
        return np.linalg.norm(hand_pos[:2] - target_pos[:2]) < self.xy_above_threshold and \
               (hand_pos[2] - target_pos[2]) > self.z_above_threshold
    
    def _check_placed(self, object_pos, target_pos):
        return np.linalg.norm(object_pos - target_pos) < self.place_threshold 
    
    def _check_released(self, hand_pos, target_pos):
        # TODO check force sensing
        return np.linalg.norm(hand_pos - target_pos) < self.grasp_threshold
    
    
    def plan(self, obs):
        ## Extract information from the observation
        joint_angles = obs['state'][:8]
        hand_pos = obs['state'][8:11]
        hand_quat = obs['state'][11:15]
        object_pos = obs['object'][:3]
        object_quat = obs['object'][3:]
        target_pos = obs["target"][:3]

        obs["grasped_pos"] = self.grasped_pos

        ## Task planning logic
        if self.state == "initial":
            if self._check_above_object(hand_pos, object_pos):
                self.state = "grasping_object"
            else:
                self.state = "move_above_object"
            action = joint_angles.copy()  # No action needed, just update state
        elif self.state == "move_above_object":
            action = self.motion_planner.move_above(joint_angles, obs)
            if self._check_above_object(hand_pos, object_pos):
                self.state = "approaching_object"
        elif self.state == "approaching_object":
            action = self.motion_planner.approach(joint_angles, obs)
            if self._check_graspable(hand_pos, object_pos):
                self.state = "grasping_object"
        elif self.state == "grasping_object":
            action = self.motion_planner.grasp(joint_angles, obs)
            if self._check_grasped(hand_pos, object_pos):
                self.state = "lifting_object"
                self.grasped_pos = object_pos.copy()
        elif self.state == "lifting_object":
            action = self.motion_planner.lift(joint_angles, obs)
            if self._check_lifted(object_pos):
                self.state = "moving_to_target"
        elif self.state == "moving_to_target":
            action = self.motion_planner.move_to_target(joint_angles, obs)
            if self._check_above_target(hand_pos, target_pos):
                self.state = "placing_object"
        elif self.state == "placing_object":
            action = self.motion_planner.place(joint_angles, obs)
            if self._check_placed(object_pos, target_pos):
                self.state = "release_object"
        elif self.state == "release_object":
            action = self.motion_planner.release(joint_angles, obs)
            if self._check_released(hand_pos, target_pos):
                self.state = "success"
        elif self.state == "success":
            action = joint_angles.copy()
        else:
            raise ValueError(f"Unknown state: {self.state}")
        return action, self.state

            


