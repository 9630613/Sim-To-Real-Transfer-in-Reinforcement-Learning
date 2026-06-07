import gym
import numpy as np
from gym import spaces, register
from stable_baselines3 import SAC
from env.custom_hopper import *
from stable_baselines3.common.evaluation import evaluate_policy
# Create environments
source_env = gym.make('CustomHopper-source-v0')
target_env = gym.make('CustomHopper-target-v0')


#load model
source_model = SAC.load("hopper_customhopper-target-v0")
source_model = SAC.load("hopper_customhopper-source-v0")
udr_model = SAC.load("hopper_customhopper-udr-v0")
adr_model = SAC.load("hopper_customhopper-adr-v0")

# Run evaluations
source_mean_reward, source_std_reward = evaluate_policy(source_model,source_env, n_eval_episodes=50,deterministic=True)  #evaluate source on source
target_mean_reward, target_std_reward = evaluate_policy(target_model, target_env,n_eval_episodes=50,deterministic=True)  #evaluate Target on target
sourceOnTarget_mean_reward, sourceOnTarget_std_reward = evaluate_policy(source_model, target_env, n_eval_episodes=50,deterministic=True)  #evaluate source on target
udrOnSource_mean_reward, udrOnSource_std_reward = evaluate_policy(udr_model, source_env, n_eval_episodes=50,deterministic=True) #evaluate udr on source
udronTarget_mean_reward,udronTarget_std_reward = evaluate_policy(udr_model, target_env, n_eval_episodes=50,deterministic=True) #evaluate udr on target
adrOnSource_mean_reward, adrOnSource_std_reward = evaluate_policy(adr_model, source_env, n_eval_episodes=50,deterministic=True) #evaluate adr on source
adronTarget_mean_reward,adronTarget_std_reward = evaluate_policy(adr_model, target_env, n_eval_episodes=50,deterministic=True) #evaluate adr on target

# Print results
print("\nSource Model on Source Domain Results:")
print(f"Mean Reward: {source_mean_reward:.2f} ± {source_std_reward:.2f}")

print("\nSource Model on Target Domain Results:")
print(f"Mean Reward: {sourceOnTarget_mean_reward:.2f} ± {sourceOnTarget_std_reward:.2f}")

print("\nTarget Model on Target Domain Results:")
print(f"Mean Reward: {target_mean_reward:.2f} ± {target_std_reward:.2f}")

print("\nUDR Model on Source Domain Results:")
print(f"Mean Reward: {udrOnSource_mean_reward:.2f} ± {udrOnSource_std_reward:.2f}")

print("\nUDR Model on Target Domain Results:")
print(f"Mean Reward: {udronTarget_mean_reward:.2f} ± {udronTarget_std_reward:.2f}")

print("\nADR Model on Source Domain Results:")
print(f"Mean Reward: {adrOnSource_mean_reward:.2f} ± {adrOnSource_std_reward:.2f}")

print("\nADR Model on Target Domain Results:")
print(f"Mean Reward: {adronTarget_mean_reward:.2f} ± {adronTarget_std_reward:.2f}")