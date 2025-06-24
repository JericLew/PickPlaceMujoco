import numpy as np
import random

class RRT():
    def __init__(self, joint_limits, step_size=0.1, max_iter=1000):
        """
        joint_limits: list of (min, max) tuples for each joint
        step_size: how far to extend towards sampled points
        max_iter: maximum number of iterations
        """
        self.joint_limits = joint_limits
        self.step_size = step_size
        self.max_iter = max_iter

    def sample(self):
        return np.array([random.uniform(lim[0], lim[1]) for lim in self.joint_limits])

    def steer(self, from_angle, to_angle):
        direction = to_angle - from_angle
        dist = np.linalg.norm(direction)
        if dist < self.step_size:
            return to_angle
        return from_angle + self.step_size * direction / dist

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
                if np.linalg.norm(new_node - target_joint_angles) < self.step_size:
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
