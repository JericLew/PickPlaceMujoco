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

            # # Check if new node is within joint limits
            # if any(new_node < np.array([lim[0] for lim in self.joint_limits])) or \
            #    any(new_node > np.array([lim[1] for lim in self.joint_limits])):
            #     print(f"New node {new_node} out of joint limits, skipping")
            #     continue

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
        print("RRT* failed to find a path after max iterations.")
        return [np.array(current_joint_angles)]

class RRTStar():
    def __init__(self, joint_limits, step_size=(0.1,), max_iter=5000, neighbor_radius=0.5):
        self.joint_limits = joint_limits
        if isinstance(step_size, (float, int)):
            self.step_size = np.array([step_size] * len(joint_limits))
        elif len(step_size) == 1 and len(joint_limits) > 1:
            self.step_size = np.array([step_size[0]] * len(joint_limits))
        else:
            self.step_size = np.array(step_size)
        self.max_iter = max_iter
        self.neighbor_radius = neighbor_radius

    def sample(self):
        return np.array([random.uniform(lim[0], lim[1]) for lim in self.joint_limits])

    def steer(self, from_angle, to_angle):
        direction = to_angle - from_angle
        step = np.clip(direction, -self.step_size, self.step_size)
        new_node = from_angle + step
        if np.all(np.abs(direction) <= self.step_size):
            return to_angle
        return new_node

    def get_nearby_indices(self, nodes, new_node):
        dists = [np.linalg.norm(n - new_node) for n in nodes]
        return [i for i, d in enumerate(dists) if d <= self.neighbor_radius]

    def get_actions(self, current_joint_angles, target_joint_angles):
        nodes = [np.array(current_joint_angles)]
        parents = [-1]
        costs = [0.0]

        for i in range(self.max_iter):
            if i % 500 == 0:
                print(f"Iteration {i}/{self.max_iter}")
            if random.random() < 0.1:
                sample = np.array(target_joint_angles)
            else:
                sample = self.sample()

            # Find nearest node
            dists = [np.linalg.norm(n - sample) for n in nodes]
            nearest_idx = np.argmin(dists)
            nearest = nodes[nearest_idx]

            new_node = self.steer(nearest, sample)

            # # Check if new node is within joint limits
            # if any(new_node < np.array([lim[0] for lim in self.joint_limits])) or \
            #    any(new_node > np.array([lim[1] for lim in self.joint_limits])):
            #     continue

            # Find neighbors within radius
            neighbor_indices = self.get_nearby_indices(nodes, new_node)
            # Choose parent with minimal cost
            min_cost = costs[nearest_idx] + np.linalg.norm(new_node - nearest)
            min_parent = nearest_idx
            for idx in neighbor_indices:
                cost = costs[idx] + np.linalg.norm(new_node - nodes[idx])
                if cost < min_cost:
                    min_cost = cost
                    min_parent = idx

            nodes.append(new_node)
            parents.append(min_parent)
            costs.append(min_cost)

            # Rewire neighbors
            new_idx = len(nodes) - 1
            for idx in neighbor_indices:
                cost_through_new = costs[new_idx] + np.linalg.norm(nodes[idx] - new_node)
                if cost_through_new < costs[idx]:
                    parents[idx] = new_idx
                    costs[idx] = cost_through_new

            # Check if goal is reached
            if np.all(np.abs(new_node - target_joint_angles) <= np.array(self.step_size)):
                # Find best goal node among all nodes close to goal
                goal_indices = [i for i, n in enumerate(nodes)
                                if np.all(np.abs(n - target_joint_angles) <= np.array(self.step_size))]
                if not goal_indices:
                    continue
                best_goal = min(goal_indices, key=lambda i: costs[i])
                # Reconstruct path
                path = [nodes[best_goal]]
                idx = best_goal
                while parents[idx] != -1:
                    idx = parents[idx]
                    path.append(nodes[idx])
                path.reverse()
                return path
            
        print("RRT* failed to find a path after max iterations.")
        return [np.array(current_joint_angles)]
