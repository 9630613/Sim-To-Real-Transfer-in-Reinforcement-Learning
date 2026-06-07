"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
import gym
from env.custom_hopper import *
import gym
import numpy as np
from gym import spaces, register
from stable_baselines3 import SAC
from env.custom_hopper import *

def main():
         

    def train_and_evaluate(env_id, total_timesteps=2000000, n_eval_episodes=50, max_episode_steps=None):
        
        # Optionally set max episode steps

        if max_episode_steps:
            env = gym.make(env_id, max_episode_steps=max_episode_steps)
        else:
            env = gym.make(env_id)
            # Print environment info
        print(f'\nTraining on environment: {env_id}')
        print('State space:', env.observation_space)
        print('Action space:', env.action_space)
        print('Dynamics parameters:', env.get_parameters())

        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=5e-4,  # Slightly higher
            buffer_size=100000,  # Reduced
            batch_size=128,      # Smaller
            learning_starts=500, # Quicker start
            train_freq=(4, "step"),  # Less frequent updates
            gradient_steps=1,    # Fewer gradient steps
            tau=0.01,
            verbose=1
            )

        # Train the model
        print("\nStarting training...")
        model.learn(total_timesteps=total_timesteps)

        # Save the model
        model_path = f"hopper_{env_id.lower()}"
        model.save(model_path)
        print(f"\nModel saved to {model_path}")


        return model


    # Train on source environment
    source_model, source_reward, source_std = train_and_evaluate(
        'CustomHopper-target-v0',
        total_timesteps=1000000,
        n_eval_episodes=50
    )

if __name__ == '__main__':
    main()