import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../OT2_Twin'))
from sim_class import Simulation


class WrappedEnv(gym.Env):
    def __init__(self, render=False, max_step=1000):
        super(WrappedEnv, self).__init__()
        self.render = render
        self.max_step = max_step

        self.x_range = [-0.187, 0.253]
        self.y_range = [-0.1705, 0.2195]
        self.z_range = [0.1195, 0.2895]
        xyz_low = np.array([self.x_range[0],
                            self.y_range[0],
                            self.z_range[0]] * 2).astype(np.float32)
        xyz_high = np.array([self.x_range[1],
                             self.y_range[1],
                             self.z_range[1]] * 2).astype(np.float32)

        self.sim = Simulation(num_agents=1, render=render)

        self.action_space = spaces.Discrete(4, )
        self.observation_space = spaces.Box(xyz_low, xyz_high, (6, ),
                                            dtype=np.float32)

        self.steps = 0

    def reset(self, seed=None):

        if seed is not None:
            np.random.seed(seed)

        self.goal_position = np.array([
            np.random.uniform(low=self.x_range[0], high=self.x_range[1]),
            np.random.uniform(low=self.y_range[0], high=self.y_range[1]),
            np.random.uniform(low=self.z_range[0], high=self.z_range[1])
        ])

        observation = self.sim.reset(num_agents=1)
        observation = np.concatenate(
            [observation[list(observation.keys())[0]]['pipette_position'],
                self.goal_position]).astype(np.float32)  # For one agent only !

        self.steps = 0

        info = {}

        return observation, info

    def step(self, action=None):

        if action is None:
            action = np.random.randint(0, 6)

        # self.action = action
        self.steps += 1

        # Map the discrete action to specific velocities
        if action == 0:
            velocity_x, velocity_y, velocity_z = 0.5, 0.0, 0.0
        elif action == 1:
            velocity_x, velocity_y, velocity_z = -0.5, 0.0, 0.0
        elif action == 2:
            velocity_x, velocity_y, velocity_z = 0.0, 0.5, 0.0
        elif action == 3:
            velocity_x, velocity_y, velocity_z = 0.0, -0.5, 0.0
        elif action == 4:
            velocity_x, velocity_y, velocity_z = 0.0, 0.0, 0.5
        elif action == 5:
            velocity_x, velocity_y, velocity_z = 0.0, 0.0, -0.5

        # Example action: Move joints with specific velocities
        drop_command = 0  # Example drop command

        actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

        obs = self.sim.run(actions)

        # obs = tuple(obs['robotId_1']['pipette_position'])+
        # tuple(self.goal_position)
        obs = np.concatenate((obs[list(obs.keys())[0]]['pipette_position'],
                              self.goal_position)).astype(np.float32)  # For one agent only !

        reward = -np.linalg.norm(np.array(obs[:3]) - np.array(obs[3:]))
        reward = float(reward)

        self.done_threshold = -0.01  # Threshold for termination

        terminated = False
        if reward > self.done_threshold:
            reward += 10

            terminated = True

        truncaded = False
        if self.steps >= self.max_step:
            truncaded = True

        info = {}

        return obs, reward, terminated, truncaded, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()
