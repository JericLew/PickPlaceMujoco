import numpy as np
import random

class RRT():
    def __init__(self, joint_limits, step_size=(0.1,), max_iter=5000):
        """
        joint_limits: list of (min, max) tuples for each joint
        step_size: tuple of step sizes for each joint
        max_iter: maximum number of iterations
        """
        self.joint_limits = joint_limits
        if isinstance(step_size, (float, int)):
            self.step_size = np.array([step_size] * len(joint_limits))
        elif len(step_size) == 1 and len(joint_limits) > 1:
            self.step_size = np.array([step_size[0]] * len(joint_limits))
        else:
            self.step_size = np.array(step_size)
        self.max_iter = max_iter

    def sample(self):
        return np.array([random.uniform(lim[0], lim[1]) for lim in self.joint_limits])

    def steer(self, from_angle, to_angle):
        direction = to_angle - from_angle
        # Scale direction by per-joint step size
        step = np.clip(direction, -self.step_size, self.step_size)
        new_node = from_angle + step
        # If all joints are within step size, just return to_angle
        if np.all(np.abs(direction) <= self.step_size):
            return to_angle
        return new_node

    def is_collision_free(self, q1, q2):
        # Placeholder: always returns True
        # Replace with actual collision checking
        return True

    def get_action(self, current_joint_angles, target_joint_angles):
        """
        Plans a path from current_joint_angles to target_joint_angles using RRT.
        Returns the next joint angles to move towards.
        """
        nodes = [np.array(current_joint_angles)]
        parents = [-1]

        for i in range(self.max_iter):
            if random.random() < 0.1:
                sample = np.array(target_joint_angles)
            else:
                sample = self.sample()

            # Find nearest node
            dists = [np.linalg.norm(n - sample) for n in nodes]
            nearest_idx = np.argmin(dists)
            nearest = nodes[nearest_idx]

            new_node = self.steer(nearest, sample)
            if self.is_collision_free(nearest, new_node):
                nodes.append(new_node)
                parents.append(nearest_idx)

                # Check if goal is reached
                if np.all(np.abs(new_node - target_joint_angles) <= np.array(self.step_size)):
                    # Reconstruct path
                    path = [new_node]
                    idx = len(nodes) - 1
                    while parents[idx] != -1:
                        idx = parents[idx]
                        path.append(nodes[idx])
                    path.reverse()
                    # Return the next step towards the goal
                    if len(path) > 1:
                        return path[1]
                    else:
                        return path[0]

        # If no path found, return current position
        return np.array(current_joint_angles)
    
    def get_actions(self, current_joint_angles, target_joint_angles):
        """
        Plans a path from current_joint_angles to target_joint_angles using RRT.
        Returns the sequence of joint angles (including start and goal) as a list.
        """
        nodes = [np.array(current_joint_angles)]
        parents = [-1]

        for i in range(self.max_iter):
            if random.random() < 0.1:
                sample = np.array(target_joint_angles)
            else:
                sample = self.sample()

            # Find nearest node
            dists = [np.linalg.norm(n - sample) for n in nodes]
            nearest_idx = np.argmin(dists)
            nearest = nodes[nearest_idx]

            new_node = self.steer(nearest, sample)
            if self.is_collision_free(nearest, new_node):
                nodes.append(new_node)
                parents.append(nearest_idx)

                # Check if goal is reached
                if np.all(np.abs(new_node - target_joint_angles) <= np.array(self.step_size)):
                    # Reconstruct path
                    path = [new_node]
                    idx = len(nodes) - 1
                    while parents[idx] != -1:
                        idx = parents[idx]
                        path.append(nodes[idx])
                    path.reverse()
                    return path  # Return the full sequence of actions

        # If no path found, return just the start position
        return [np.array(current_joint_angles)]
