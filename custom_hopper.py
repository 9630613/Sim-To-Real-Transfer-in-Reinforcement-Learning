"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm
from collections import deque

class CustomHopper(MujocoEnv, utils.EzPickle):
    
    def __init__(self, domain=None):

        self.episode_reward = 0.0
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)
        
        self.domain = domain
        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
         

        # ADR-specific attributes
        self.performance_history = deque(maxlen=100)  # Store last 100 episode rewards
        

        # Initialize mass ranges for ADR
        self.mass_ranges = {
            'thigh': {'min': 0.8, 'max': 1.2, 'step': 0.05},  # ±20% initially
            'leg': {'min': 0.8, 'max': 1.2, 'step': 0.05},
            'foot': {'min': 0.8, 'max': 1.2, 'step': 0.05}
        }
            
        if domain == 'source':  
            self.sim.model.body_mass[1] -= 1.0
        elif domain == 'adr':
            self.set_random_parameters()
            self.sim.model.body_mass[1] -= 1.0
        elif domain == 'udr':
            self.set_random_parameters()
            self.sim.model.body_mass[1] -= 1.0
            
    
    def adjust_ranges(self):
        """Adjust mass ranges based on recent performance"""
        if len(self.performance_history) < self.performance_history.maxlen:
            return

        avg_performance = np.mean(self.performance_history)
        
        # Thresholds for range adjustment
        expansion_threshold = 200  # Good performance
        contraction_threshold = 100  # Poor performance
        
        for part in self.mass_ranges:
            if avg_performance > expansion_threshold:
                # Expand ranges if doing well
                self.mass_ranges[part]['min'] = max(
                    self.mass_ranges[part]['min'] - self.mass_ranges[part]['step'],
                    0.5  # Minimum 50% of original mass
                )
                self.mass_ranges[part]['max'] = min(
                    self.mass_ranges[part]['max'] + self.mass_ranges[part]['step'],
                    1.5  # Maximum 150% of original mass
                )
            elif avg_performance < contraction_threshold:
                # Contract ranges if doing poorly
                new_min = self.mass_ranges[part]['min'] + self.mass_ranges[part]['step']
                new_max = self.mass_ranges[part]['max'] - self.mass_ranges[part]['step']
                # Ensure minimum range of 20% is maintained
                if new_max - new_min >= 0.2:
                    self.mass_ranges[part]['min'] = new_min
                    self.mass_ranges[part]['max'] = new_max       


    def set_random_parameters(self):
        """Set random masses using current ranges"""
        if self.domain == 'udr':
            # Original UDR behavior
            random_masses = self.sample_parameters()
        elif self.domain == 'adr':
            # ADR behavior
            random_masses = self.sample_adr_parameters()
        else:
            return
        
        self.set_parameters(random_masses)

    def sample_adr_parameters(self):
        """Sample masses according to current ADR ranges"""
        masses = np.array(self.original_masses)
        randomized_masses = masses.copy()
        
        # Map array indices to mass_ranges keys
        part_map = {1: 'thigh', 2: 'leg', 3: 'foot'}
        
        # Randomize all masses except torso (index 0)
        for i in range(1, len(masses)):
            if i in part_map:
                ranges = self.mass_ranges[part_map[i]]
                base_mass = masses[i]
                randomized_masses[i] = base_mass * np.random.uniform(
                    ranges['min'],
                    ranges['max']
                )
        
        return randomized_masses

    def sample_parameters(self):
        """Original UDR sampling"""
        masses = np.array(self.original_masses)
        randomized_masses = masses.copy()
        
        # Randomize all masses except torso (index 0)
        for i in range(1, len(masses)):
            base_mass = masses[i]
            randomized_masses[i] = np.random.uniform(
                base_mass * 0.8,  # -20%
                base_mass * 1.2   # +20%
            )
        return randomized_masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        
        # Track episode reward for ADR
        self.episode_reward = self.episode_reward + reward

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))

        # If episode is done, update ADR statistics
        if done and self.domain == 'adr':
            self.performance_history.append(self.episode_reward)
            self.adjust_ranges()

        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.domain in ['udr', 'adr']:
            self.set_random_parameters()
        # Reset episode reward tracker
        self.episode_reward = 0.0
        
        return ob

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.episode_reward = 0.0 
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

gym.envs.register(
        id="CustomHopper-udr-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "udr"}
)

gym.envs.register(
        id="CustomHopper-adr-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "adr"}
)