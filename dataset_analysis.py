import argparse
import pickle
import numpy as np
from d4rl import get_normalized_score




parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='hopper')
parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert

args = parser.parse_args()
dataset_path = f'data/{args.env}-{args.dataset}-v2.pkl'

with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)

states, traj_lens, returns = [], [], []
for path in trajectories:
    states.append(path['observations'])
    traj_lens.append(len(path['observations']))
    returns.append(path['rewards'].sum())

traj_lens, returns = np.array(traj_lens), np.array(returns)  

print(f'Number of trajectories: {len(traj_lens)}')
print(f'Average return: {np.mean(returns)}, std: {np.std(returns)}, max: {np.max(returns)}, min: {np.min(returns)}')
print(f'Average trajectory length: {np.mean(traj_lens)}, std: {np.std(traj_lens)}, max: {np.max(traj_lens)}, min: {np.min(traj_lens)}')

#Normalised scores
env_name_str = f'{args.env}-{args.dataset}-v2'
mean_norm_score  = get_normalized_score(env_name_str,np.mean(returns)) * 100
max_norm_score  = get_normalized_score(env_name_str,np.max(returns)) * 100
min_norm_score  = get_normalized_score(env_name_str,np.min(returns)) * 100
std_norm_score  = get_normalized_score(env_name_str,np.std(returns)) * 100


print(f'Normalized mean score: {mean_norm_score}')
print(f'Normalized max score: {max_norm_score}')
print(f'Normalized min score: {min_norm_score}')
print(f'Normalized std score: {std_norm_score}')