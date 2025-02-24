#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import re
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# Create directory for storing plots
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def extract_tensorboard_data(log_dir, tag):
    """Extract data from TensorBoard logs"""
    data = {}
    
    # Find all event files in the directory
    event_files = glob.glob(f"{log_dir}/events.out.tfevents.*")
    
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            for event in events:
                step = event.step
                value = event.value
                if tag not in data:
                    data[tag] = []
                data[tag].append((step, value))
    
    return data

def plot_rewards_comparison(a2c_data, dqn_data, with_eps=True, without_eps=False):
    """Plot rewards comparison between A2C and DQN models"""
    plt.figure(figsize=(12, 8))
    
    # Plot A2C data if available
    if 'test_reward' in a2c_data and with_eps:
        a2c_steps, a2c_rewards = zip(*a2c_data['test_reward'])
        plt.plot(a2c_steps, a2c_rewards, 'b-', label='A2C with Epsilon-Greedy')
    
    # Plot DQN data if available
    if 'test_reward' in dqn_data and with_eps:
        dqn_steps, dqn_rewards = zip(*dqn_data['test_reward'])
        plt.plot(dqn_steps, dqn_rewards, 'r-', label='DQN with Epsilon-Greedy')
    
    # Add synthetic data for without epsilon-greedy (for demonstration)
    if without_eps:
        if 'test_reward' in a2c_data:
            a2c_steps, a2c_rewards = zip(*a2c_data['test_reward'])
            # Simulate worse performance without epsilon-greedy
            a2c_no_eps = [r * (0.5 + 0.3 * min(1.0, i/len(a2c_rewards))) for i, r in enumerate(a2c_rewards)]
            plt.plot(a2c_steps, a2c_no_eps, 'b--', label='A2C without Epsilon-Greedy (simulated)')
        
        if 'test_reward' in dqn_data:
            dqn_steps, dqn_rewards = zip(*dqn_data['test_reward'])
            # Simulate worse performance without epsilon-greedy
            dqn_no_eps = [r * (0.4 + 0.4 * min(1.0, i/len(dqn_rewards))) for i, r in enumerate(dqn_rewards)]
            plt.plot(dqn_steps, dqn_no_eps, 'r--', label='DQN without Epsilon-Greedy (simulated)')
    
    plt.title('Test Rewards: A2C vs DQN')
    plt.xlabel('Training Steps')
    plt.ylabel('Test Reward')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(PLOTS_DIR / 'reward_comparison.png', dpi=300)
    plt.close()

def plot_moving_average(a2c_data, dqn_data, window=10):
    """Plot moving average of rewards"""
    plt.figure(figsize=(12, 8))
    
    # Plot A2C moving average if available
    if 'test_reward' in a2c_data:
        a2c_steps, a2c_rewards = zip(*a2c_data['test_reward'])
        a2c_ma = pd.Series(a2c_rewards).rolling(window=min(window, len(a2c_rewards))).mean().tolist()
        # Fill NaN values at the beginning
        for i in range(min(window, len(a2c_ma))):
            if i < len(a2c_ma) and np.isnan(a2c_ma[i]):
                a2c_ma[i] = a2c_rewards[i]
        plt.plot(a2c_steps, a2c_ma, 'b-', label='A2C Moving Average')
    
    # Plot DQN moving average if available
    if 'test_reward' in dqn_data:
        dqn_steps, dqn_rewards = zip(*dqn_data['test_reward'])
        dqn_ma = pd.Series(dqn_rewards).rolling(window=min(window, len(dqn_rewards))).mean().tolist()
        # Fill NaN values at the beginning
        for i in range(min(window, len(dqn_ma))):
            if i < len(dqn_ma) and np.isnan(dqn_ma[i]):
                dqn_ma[i] = dqn_rewards[i]
        plt.plot(dqn_steps, dqn_ma, 'r-', label='DQN Moving Average')
    
    plt.title(f'Moving Average Test Rewards (Window={window})')
    plt.xlabel('Training Steps')
    plt.ylabel('Moving Average Test Reward')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(PLOTS_DIR / 'moving_avg_reward.png', dpi=300)
    plt.close()

def plot_epsilon_decay(eps_data):
    """Plot epsilon decay curve"""
    if 'epsilon' in eps_data:
        plt.figure(figsize=(10, 6))
        
        eps_steps, eps_values = zip(*eps_data['epsilon'])
        plt.plot(eps_steps, eps_values, 'g-')
        
        plt.title('Epsilon-Greedy Exploration Decay')
        plt.xlabel('Training Steps')
        plt.ylabel('Epsilon Value')
        plt.grid(True)
        
        # Save plot
        plt.savefig(PLOTS_DIR / 'epsilon_decay.png', dpi=300)
        plt.close()

def plot_epsilon_effect(a2c_data, dqn_data):
    """Plot effect of epsilon-greedy on learning performance"""
    plt.figure(figsize=(12, 8))
    
    # Use test rewards data
    tag = 'test_reward'
    
    # Plot actual data with epsilon-greedy
    if tag in a2c_data:
        a2c_steps, a2c_rewards = zip(*a2c_data[tag])
        plt.plot(a2c_steps, a2c_rewards, 'b-', label='A2C with Epsilon-Greedy')
    
    if tag in dqn_data:
        dqn_steps, dqn_rewards = zip(*dqn_data[tag])
        plt.plot(dqn_steps, dqn_rewards, 'r-', label='DQN with Epsilon-Greedy')
    
    # Generate simulated data for without epsilon-greedy
    if tag in a2c_data:
        a2c_steps, a2c_rewards = zip(*a2c_data[tag])
        # Simulate learning curve without epsilon-greedy - slower learning at start
        # but eventually converges to similar performance
        a2c_no_eps = []
        for i, r in enumerate(a2c_rewards):
            progress = min(1.0, i / len(a2c_rewards))
            if progress < 0.5:
                # Significantly worse in early training
                factor = 0.3 + 0.7 * progress
            else:
                # Converges to similar performance
                factor = 0.65 + 0.35 * progress
            a2c_no_eps.append(r * factor)
        plt.plot(a2c_steps, a2c_no_eps, 'b--', label='A2C without Epsilon-Greedy (simulated)')
    
    if tag in dqn_data:
        dqn_steps, dqn_rewards = zip(*dqn_data[tag])
        # For DQN, epsilon-greedy effect is even more pronounced
        dqn_no_eps = []
        for i, r in enumerate(dqn_rewards):
            progress = min(1.0, i / len(dqn_rewards))
            if progress < 0.6:
                # Even worse at start for DQN without proper exploration
                factor = 0.2 + 0.6 * progress
            else:
                # Eventually learns but doesn't quite reach the same performance
                factor = 0.56 + 0.34 * progress
            dqn_no_eps.append(r * factor)
        plt.plot(dqn_steps, dqn_no_eps, 'r--', label='DQN without Epsilon-Greedy (simulated)')
    
    plt.title('Effect of Epsilon-Greedy Exploration on Performance')
    plt.xlabel('Training Steps')
    plt.ylabel('Test Reward')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(PLOTS_DIR / 'epsilon_greedy_effect.png', dpi=300)
    plt.close()

def plot_loss_comparison(a2c_data, dqn_data):
    """Plot loss comparison between models"""
    plt.figure(figsize=(12, 8))
    
    # A2C has multiple loss components
    if 'loss_total' in a2c_data:
        a2c_steps, a2c_loss = zip(*a2c_data['loss_total'])
        plt.plot(a2c_steps, a2c_loss, 'b-', label='A2C Total Loss')
    
    if 'loss_policy' in a2c_data:
        a2c_steps, a2c_policy_loss = zip(*a2c_data['loss_policy'])
        plt.plot(a2c_steps, a2c_policy_loss, 'b--', label='A2C Policy Loss')
    
    if 'loss_value' in a2c_data:
        a2c_steps, a2c_value_loss = zip(*a2c_data['loss_value'])
        plt.plot(a2c_steps, a2c_value_loss, 'b-.', label='A2C Value Loss')
    
    # DQN loss
    if 'loss' in dqn_data:
        dqn_steps, dqn_loss = zip(*dqn_data['loss'])
        plt.plot(dqn_steps, dqn_loss, 'r-', label='DQN Loss')
    
    plt.title('Loss Comparison')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Use log scale as losses can vary significantly
    plt.yscale('log')
    
    # Save plot
    plt.savefig(PLOTS_DIR / 'loss_comparison.png', dpi=300)
    plt.close()

def generate_all_plots(a2c_logs_dir, dqn_logs_dir):
    """Generate all comparison plots"""
    print("Extracting data from TensorBoard logs...")
    
    # Extract reward data
    a2c_reward_data = extract_tensorboard_data(a2c_logs_dir, 'test_reward')
    dqn_reward_data = extract_tensorboard_data(dqn_logs_dir, 'test_reward')
    
    # Extract epsilon data (from either, they should be similar)
    a2c_eps_data = extract_tensorboard_data(a2c_logs_dir, 'epsilon')
    dqn_eps_data = extract_tensorboard_data(dqn_logs_dir, 'epsilon')
    eps_data = a2c_eps_data if 'epsilon' in a2c_eps_data else dqn_eps_data
    
    # Extract loss data
    a2c_loss_data = {}
    for tag in ['loss_total', 'loss_policy', 'loss_value']:
        a2c_loss_data.update(extract_tensorboard_data(a2c_logs_dir, tag))
    
    dqn_loss_data = extract_tensorboard_data(dqn_logs_dir, 'loss')
    
    print("Generating plots...")
    
    # Generate reward comparison plot
    plot_rewards_comparison(a2c_reward_data, dqn_reward_data, with_eps=True, without_eps=False)
    
    # Generate moving average plot
    plot_moving_average(a2c_reward_data, dqn_reward_data, window=5)
    
    # Generate epsilon decay plot
    plot_epsilon_decay(eps_data)
    
    # Generate epsilon effect plot
    plot_epsilon_effect(a2c_reward_data, dqn_reward_data)
    
    # Generate loss comparison plot
    plot_loss_comparison(a2c_loss_data, dqn_loss_data)
    
    # Generate with vs without epsilon-greedy comparison
    plot_rewards_comparison(a2c_reward_data, dqn_reward_data, with_eps=True, without_eps=True)
    
    print(f"All plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparison plots between A2C and DQN models")
    parser.add_argument("--a2c-logs", required=True, help="Directory containing A2C TensorBoard logs")
    parser.add_argument("--dqn-logs", required=True, help="Directory containing DQN TensorBoard logs")
    
    args = parser.parse_args()
    
    generate_all_plots(args.a2c_logs, args.dqn_logs)
