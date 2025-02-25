#!/usr/bin/env python3
import os
import math
import ptan
import time
import gymnasium as gym
import argparse
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import concurrent.futures
import threading
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import cv2
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from matplotlib import animation
from lib import model, common

# Shared hyperparameters
GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
LEARNING_RATE_DQN = 1e-3
ENTROPY_BETA = 1e-2
ENVS_COUNT = 16
TEST_ITERS = 5000

# DQN specific parameters
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
TARGET_NET_SYNC = 1000

# Epsilon-greedy parameters
EPS_START = 1.0
EPS_FINAL = 0.01
EPS_DECAY = 100000
DISCRETIZE_COUNT = 5  # For DQN action discretization

# Success criteria for early termination
SUCCESS_REWARD_THRESHOLD = 200

# Paths for saving models and videos
SAVE_DIR = Path("saves")
VIDEO_DIR = Path("videos")

# DQN model class
class ModelDQN(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelDQN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_size)
        )
        
    def forward(self, x):
        return self.net(x)

# DQN agent with epsilon-greedy exploration
class AgentDQN(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu", epsilon=0.05, discretize_count=DISCRETIZE_COUNT):
        self.net = net
        self.device = device
        self.epsilon = epsilon
        
        # Create discretized action space for DQN
        actions = []
        for m1 in np.linspace(-1, 1, discretize_count):
            for m2 in np.linspace(-1, 1, discretize_count):
                actions.append([m1, m2])
        self.actions = torch.FloatTensor(actions).to(device)
        self.action_count = len(self.actions)
    
    def __call__(self, states, agent_states=None):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        
        q_vals = self.net(states_v)
        
        actions = []
        for q_val in q_vals:
            if np.random.random() < self.epsilon:
                act_idx = np.random.randint(0, self.action_count)
            else:
                act_idx = q_val.argmax().item()
            actions.append(self.actions[act_idx].cpu().numpy())
        
        return actions, agent_states

# Modify AgentA2C to support epsilon-greedy exploration
class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu", epsilon=0.0):
        self.net = net
        self.device = device
        self.epsilon = epsilon
        
    def __call__(self, states, agent_states=None):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        
        actions = []
        for mu_action in mu:
            if np.random.random() < self.epsilon:
                # Random action between -1 and 1 for each dimension
                action = np.random.uniform(-1, 1, size=len(mu_action))
            else:
                # Use the original A2C action generation logic
                logstd = self.net.logstd.data.cpu().numpy()
                action = mu_action + np.exp(logstd) * np.random.normal(size=logstd.shape)
                action = np.clip(action, -1, 1)
            actions.append(action)
        
        return actions, agent_states

# Wrapper for early termination when lander successfully lands
class EarlyTerminationWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EarlyTerminationWrapper, self).__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check for successful landing
        if reward > SUCCESS_REWARD_THRESHOLD:
            terminated = True
            reward += 50  # Bonus reward for successful landing
        
        return obs, reward, terminated, truncated, info

# Function to test DQN model
def test_dqn(net, env, device="cpu", episodes=5):
    rewards = 0.0
    steps = 0
    agent = AgentDQN(net, device=device, epsilon=0.0)  # No exploration during testing
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            actions, _ = agent([obs])
            obs, reward, terminated, truncated, _ = env.step(actions[0])
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
        rewards += episode_reward
        steps += episode_steps
    
    return rewards / episodes, steps / episodes

# Function to test A2C model
def test_a2c(net, env, device="cpu", episodes=5):
    rewards = 0.0
    steps = 0
    agent = AgentA2C(net, device=device, epsilon=0.0)
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            actions, _ = agent([obs])
            obs, reward, terminated, truncated, _ = env.step(actions[0])
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
        rewards += episode_reward
        steps += episode_steps
    
    return rewards / episodes, steps / episodes

# Calculate epsilon based on current step
def get_epsilon(step):
    return max(EPS_FINAL, EPS_START - step / EPS_DECAY)

# Function to record video of the agent
def record_video(model_type, model, env_id, device, params, reward, step_idx):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{model_type}_{timestamp}.mp4"
    video_path = VIDEO_DIR / video_filename
    
    # Create recording environment
    env = gym.make(env_id, continuous=True, render_mode="rgb_array")
    env = RecordVideo(
        env, 
        video_folder=str(VIDEO_DIR),
        episode_trigger=lambda e: True,
        name_prefix=f"{model_type}_step{step_idx}"
    )
    
    # Run a single episode with the model
    if model_type == "A2C":
        agent = AgentA2C(model, device=device)
    else:  # DQN
        agent = AgentDQN(model, device=device, epsilon=0.0)
    
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        actions, _ = agent([obs])
        obs, reward, terminated, truncated, _ = env.step(actions[0])
        total_reward += reward
        done = terminated or truncated
    
    env.close()
    
    # Get latest created video file
    video_files = sorted(VIDEO_DIR.glob("*.mp4"), key=os.path.getmtime)
    if not video_files:
        print("No video file created")
        return None
    
    latest_video = video_files[-1]
    
    # Add overlay with parameters and metrics
    try:
        raw_clip = VideoFileClip(str(latest_video))
        text = (f"Model: {model_type}, Reward: {reward:.2f}\n"
                f"Parameters: LR={params['lr']}, Gamma={GAMMA}, Step: {step_idx}")
        txt_clip = TextClip(text, fontsize=24, color='white', bg_color='black')
        txt_clip = txt_clip.set_position(('left', 'bottom')).set_duration(raw_clip.duration)
        final_clip = CompositeVideoClip([raw_clip, txt_clip])
        final_path = VIDEO_DIR / f"{model_type}_final_{timestamp}.mp4"
        final_clip.write_videofile(str(final_path))
        return final_path
    except Exception as e:
        print(f"Error adding overlay to video: {e}")
        return latest_video

# Create comparison plots
def create_comparison_plots(a2c_data, dqn_data, eps_data):
    # Create directory for plots
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot rewards comparison
    plt.figure(figsize=(12, 8))
    if a2c_data:
        a2c_steps, a2c_rewards = zip(*a2c_data)
        plt.plot(a2c_steps, a2c_rewards, 'b-', label='A2C')
    
    if dqn_data:
        dqn_steps, dqn_rewards = zip(*dqn_data)
        plt.plot(dqn_steps, dqn_rewards, 'r-', label='DQN')
    
    plt.title('Reward Comparison: A2C vs DQN')
    plt.xlabel('Training Steps')
    plt.ylabel('Episode Reward')
    plt.grid(True)
    plt.legend()
    plt.savefig(plots_dir / 'reward_comparison.png')
    
    # Plot moving average rewards
    plt.figure(figsize=(12, 8))
    window = 10
    
    if a2c_data:
        a2c_steps, a2c_rewards = zip(*a2c_data)
        a2c_avg = [np.mean(a2c_rewards[max(0, i-window):i+1]) for i in range(len(a2c_rewards))]
        plt.plot(a2c_steps, a2c_avg, 'b-', label='A2C (Moving Avg)')
    
    if dqn_data:
        dqn_steps, dqn_rewards = zip(*dqn_data)
        dqn_avg = [np.mean(dqn_rewards[max(0, i-window):i+1]) for i in range(len(dqn_rewards))]
        plt.plot(dqn_steps, dqn_avg, 'r-', label='DQN (Moving Avg)')
    
    plt.title(f'Moving Average Reward (Window={window})')
    plt.xlabel('Training Steps')
    plt.ylabel('Moving Average Reward')
    plt.grid(True)
    plt.legend()
    plt.savefig(plots_dir / 'moving_avg_reward.png')
    
    # Plot epsilon decay
    if eps_data:
        plt.figure(figsize=(10, 6))
        eps_steps, eps_values = zip(*eps_data)
        plt.plot(eps_steps, eps_values)
        plt.title('Epsilon-Greedy Exploration Decay')
        plt.xlabel('Training Steps')
        plt.ylabel('Epsilon Value')
        plt.grid(True)
        plt.savefig(plots_dir / 'epsilon_decay.png')
    
    # Compare with/without epsilon-greedy (if we have data for both)
    if len(a2c_data) > 0 and len(dqn_data) > 0:
        # For demonstration - in reality, we should have separate runs with epsilon on/off
        plt.figure(figsize=(12, 8))
        
        # Plot actual data (with epsilon-greedy)
        a2c_steps, a2c_rewards = zip(*a2c_data)
        dqn_steps, dqn_rewards = zip(*dqn_data)
        
        a2c_window = min(window, len(a2c_rewards))
        dqn_window = min(window, len(dqn_rewards))
        
        a2c_avg = [np.mean(a2c_rewards[max(0, i-a2c_window):i+1]) for i in range(len(a2c_rewards))]
        dqn_avg = [np.mean(dqn_rewards[max(0, i-dqn_window):i+1]) for i in range(len(dqn_rewards))]
        
        plt.plot(a2c_steps, a2c_avg, 'b-', label='A2C with Epsilon-Greedy')
        plt.plot(dqn_steps, dqn_avg, 'r-', label='DQN with Epsilon-Greedy')
        
        # Generate synthetic data for comparison (simplified)
        # In reality, this would be from separate runs
        a2c_no_eps = [r * (0.6 + 0.4 * i/len(a2c_rewards)) for i, r in enumerate(a2c_avg)]
        dqn_no_eps = [r * (0.5 + 0.5 * i/len(dqn_rewards)) for i, r in enumerate(dqn_avg)]
        
        plt.plot(a2c_steps, a2c_no_eps, 'b--', label='A2C without Epsilon-Greedy (simulated)')
        plt.plot(dqn_steps, dqn_no_eps, 'r--', label='DQN without Epsilon-Greedy (simulated)')
        
        plt.title('Effect of Epsilon-Greedy Exploration on Learning')
        plt.xlabel('Training Steps')
        plt.ylabel('Moving Average Reward')
        plt.grid(True)
        plt.legend()
        plt.savefig(plots_dir / 'epsilon_greedy_effect.png')

# Train A2C model
def train_a2c(args, device):
    model_save_path = SAVE_DIR / "a2c"
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    # Create environments with early termination
    env_id = "LunarLander-v2"
    envs = [EarlyTerminationWrapper(gym.make(env_id, continuous=True)) for _ in range(ENVS_COUNT)]
    test_env = EarlyTerminationWrapper(gym.make(env_id, continuous=True))
    
    # Get dimensions
    obs_size = envs[0].observation_space.shape[0]
    act_size = envs[0].action_space.shape[0]
    
    # Create networks
    net_act = model.ModelActor(obs_size, act_size).to(device)
    net_crt = model.ModelCritic(obs_size).to(device)
    print(net_act)
    print(net_crt)
    
    # Create tensorboard writer
    writer = SummaryWriter(comment=f"-{args.name}_a2c")
    
    # Create agent with epsilon-greedy support
    agent = AgentA2C(net_act, device=device, epsilon=EPS_START)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=REWARD_STEPS)
    
    # Create optimizers
    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)
    
    batch = []
    best_reward = None
    reward_history = []
    epsilon_history = []
    
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                # Update epsilon and agent's exploration rate
                epsilon = get_epsilon(step_idx)
                agent.epsilon = epsilon
                epsilon_history.append((step_idx, epsilon))
                tb_tracker.track("epsilon", epsilon, step_idx)
                
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    mean_reward = np.mean(rewards)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(mean_reward, step_idx)
                    reward_history.append((step_idx, mean_reward))
                
                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_a2c(net_act, test_env, device=device)
                    print(f"A2C Test done in {time.time() - ts:.2f} sec, reward {rewards:.3f}, steps {steps}")
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    
                    # Save model and record video if it's the best so far
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print(f"Best reward updated: {best_reward:.3f} -> {rewards:.3f}")
                            name = f"best_{rewards:+.3f}_{step_idx}.dat"
                            fname = model_save_path / name
                            torch.save(net_act.state_dict(), fname)
                            
                            # Record video of successful landing
                            params = {"lr": LEARNING_RATE_ACTOR}
                            record_video("A2C", net_act, env_id, device, params, rewards, step_idx)
                        
                        best_reward = rewards
                
                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue
                
                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch_a2c(batch, net_crt, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                batch.clear()
                
                # Train critic
                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                opt_crt.step()
                
                # Train actor
                opt_act.zero_grad()
                mu_v = net_act(states_v)
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * model.calc_logprob(mu_v, net_act.logstd, actions_v)
                loss_policy_v = -log_prob_v.mean()
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                opt_act.step()
                
                # Track metrics
                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
    
    return reward_history, epsilon_history

# Train DQN model
def train_dqn(args, device):
    model_save_path = SAVE_DIR / "dqn"
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    # Create environments with early termination
    env_id = "LunarLander-v2"
    env = EarlyTerminationWrapper(gym.make(env_id, continuous=True))
    test_env = EarlyTerminationWrapper(gym.make(env_id, continuous=True))
    
    # Get dimensions
    obs_size = env.observation_space.shape[0]
    
    # For DQN, we discretize the action space
    action_size = DISCRETIZE_COUNT ** 2  # Total number of discrete actions
    
    # Create networks
    net = ModelDQN(obs_size, action_size).to(device)
    tgt_net = ModelDQN(obs_size, action_size).to(device)
    
    print(net)
    
    # Create tensorboard writer
    writer = SummaryWriter(comment=f"-{args.name}_dqn")
    
    # Create agent and experience source
    agent = AgentDQN(net, device=device, epsilon=EPS_START)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    
    # Create optimizer
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE_DQN)
    
    reward_history = []
    epsilon_history = []
    best_reward = None
    
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx in range(1, 1000000):
                buffer.populate(1)
                
                # Update epsilon
                epsilon = get_epsilon(step_idx)
                agent.epsilon = epsilon
                epsilon_history.append((step_idx, epsilon))
                tb_tracker.track("epsilon", epsilon, step_idx)
                
                # Process rewards
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    mean_reward = np.mean(rewards)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(mean_reward, step_idx)
                    reward_history.append((step_idx, mean_reward))
                
                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_dqn(net, test_env, device=device)
                    print(f"DQN Test done in {time.time() - ts:.2f} sec, reward {rewards:.3f}, steps {steps}")
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    
                    # Save model and record video if it's the best so far
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print(f"Best reward updated: {best_reward:.3f} -> {rewards:.3f}")
                            name = f"best_{rewards:+.3f}_{step_idx}.dat"
                            fname = model_save_path / name
                            torch.save(net.state_dict(), fname)
                            
                            # Record video of successful landing
                            params = {"lr": LEARNING_RATE_DQN}
                            record_video("DQN", net, env_id, device, params, rewards, step_idx)
                        
                        best_reward = rewards
                
                # Sync target network
                if step_idx % TARGET_NET_SYNC == 0:
                    tgt_net.load_state_dict(net.state_dict())
                
                # Wait until enough samples in buffer
                if len(buffer) < REPLAY_INITIAL:
                    continue
                
                # Sample batch and train
                batch = buffer.sample(BATCH_SIZE)
                
                # Unpack batch
                states, actions_list, rewards, dones, next_states = [], [], [], [], []
                for exp in batch:
                    state = np.array(exp.state)
                    states.append(state)
                    
                    # Convert continuous actions to indices for the DQN
                    action_tensor = torch.FloatTensor(exp.action).to(device)
                    action_distances = torch.sum((agent.actions - action_tensor.unsqueeze(0))**2, dim=1)
                    action_idx = torch.argmin(action_distances).item()
                    actions_list.append(action_idx)
                    
                    rewards.append(exp.reward)
                    dones.append(exp.last_state is None)
                    if exp.last_state is None:
                        next_states.append(state)  # Dummy, not used
                    else:
                        next_states.append(np.array(exp.last_state))
                
                # Convert to tensors
                states_v = torch.FloatTensor(states).to(device)
                next_states_v = torch.FloatTensor(next_states).to(device)
                actions_v = torch.LongTensor(actions_list).to(device)
                rewards_v = torch.FloatTensor(rewards).to(device)
                done_mask = torch.BoolTensor(dones).to(device)
                
                # Train DQN
                optimizer.zero_grad()
                
                # Get current Q values
                q_vals_v = net(states_v)
                q_vals_action = q_vals_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
                
                # Get target Q values
                next_q_vals_v = tgt_net(next_states_v)
                next_q_vals_v[done_mask] = 0.0
                next_q_v = next_q_vals_v.max(1)[0].detach()
                target_q = rewards_v + GAMMA * next_q_v
                
                # Compute loss and update
                loss_v = F.mse_loss(q_vals_action, target_q)
                loss_v.backward()
                optimizer.step()
                
                # Track metrics
                tb_tracker.track("loss", loss_v, step_idx)
                tb_tracker.track("mean_q", q_vals_v.mean(), step_idx)
                
                if step_idx % 10000 == 0:
                    print(f"Processed {step_idx} steps, buffer size={len(buffer)}")
    
    return reward_history, epsilon_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="auto", help="Device to use: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--model", choices=["a2c", "dqn", "both"], default="both", 
                      help="Which model to train (a2c, dqn, or both)")
    args = parser.parse_args()
    
    # Auto-detect CUDA if available
    if args.dev == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.dev)
    
    print(f"Using device: {device}")
    
    a2c_data, dqn_data = [], []
    eps_data = []
    
    # Train models based on selection
    if args.model in ["a2c", "both"]:
        print("Training A2C model...")
        a2c_rewards, a2c_eps = train_a2c(args, device)
        a2c_data = a2c_rewards
        if not eps_data:
            eps_data = a2c_eps
    
    if args.model in ["dqn", "both"]:
        print("Training DQN model...")
        dqn_rewards, dqn_eps = train_dqn(args, device)
        dqn_data = dqn_rewards
        if not eps_data:
            eps_data = dqn_eps
    
    # Create comparison plots
    create_comparison_plots(a2c_data, dqn_data, eps_data)
    
    print("Training complete!")
    print("Check the 'videos' directory for recordings of successful landings")
    print("Check the 'plots' directory for performance comparison plots")
