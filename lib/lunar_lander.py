#!/usr/bin/env python3
import os
import math
import ptan
import time
import gymnasium as gym
import argparse
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo, TransformObservation
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
from collections import deque
from lib import model, common

# Force GPU detection
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available, using CPU")

# Optimized hyperparameters based on research findings
GAMMA = 0.995  # Increased discount factor for longer-term rewards
REWARD_STEPS = 5  # N-step return (better than 2-step)
BATCH_SIZE = 128  # Larger batch size for more stable gradients
LEARNING_RATE_ACTOR = 0.001  # Increased learning rate for faster convergence
LEARNING_RATE_CRITIC = 0.001 # Increased learning rate for faster convergence
LEARNING_RATE_DQN = 0.001  # DQN learning rate
ENTROPY_BETA = 0.01  # Good balance of exploration vs exploitation
ENVS_COUNT = 16
TEST_ITERS = 2000  # Test more frequently

# DQN specific improved parameters
REPLAY_SIZE = 100000
REPLAY_INITIAL = 5000  # Reduced initial replay size for faster learning
TARGET_NET_SYNC = 500  # More frequent target network updates

# Epsilon-greedy parameters (faster decay schedule)
EPS_START = 1.0
EPS_FINAL = 0.01
EPS_DECAY = 50000  # Faster decay
DISCRETIZE_COUNT = 6  # Set to 6 for consistent 36 actions

# Success criteria for early termination
SUCCESS_REWARD_THRESHOLD = 100 # 100 is usually good enough for LunarLander

# Paths for saving models and videos
SAVE_DIR = Path("saves")
VIDEO_DIR = Path("videos")

# PPO-style clipping parameter
PPO_EPS = 0.2

# Generalized Advantage Estimation parameter
GAE_LAMBDA = 0.95

# Multi-step bootstrap parameter for DQN
N_STEPS = 3

# Create a state normalizer wrapper to improve training stability
class StateNormalizer(gym.Wrapper):
    def __init__(self, env):
        super(StateNormalizer, self).__init__(env)
        self.state_mean = np.zeros(env.observation_space.shape[0])
        self.state_std = np.ones(env.observation_space.shape[0])
        self.alpha = 0.001  # Slow moving average for normalization
        self.num_steps = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._normalize_state(obs)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._normalize_state(obs)
        return obs, info
    
    def _normalize_state(self, state):
        # Update running statistics
        self.num_steps += 1
        self.state_mean = self.state_mean + self.alpha * (state - self.state_mean)
        self.state_std = self.state_std + self.alpha * ((state - self.state_mean)**2 - self.state_std)
        
        # Avoid division by zero
        norm_state = (state - self.state_mean) / (np.sqrt(self.state_std) + 1e-8)
        return norm_state

# Enhanced reward shaping wrapper to improve learning signal
class RewardShaper(gym.Wrapper):
    def __init__(self, env, landing_bonus=50.0, stability_factor=0.1, velocity_penalty=0.1):
        super(RewardShaper, self).__init__(env)
        self.landing_bonus = landing_bonus
        self.stability_factor = stability_factor
        self.velocity_penalty = velocity_penalty
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract state information relevant to landing
        # LunarLander states: [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1_contact, leg2_contact]
        x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1, leg2 = obs
        
        # Penalize excessive velocity (both horizontal and vertical)
        velocity_penalty = self.velocity_penalty * (abs(x_vel) + abs(y_vel))
        
        # Reward stability (low angular velocity and angle close to zero)
        stability_reward = self.stability_factor * (1.0 - min(1.0, abs(angle) + abs(angular_vel)))
        
        # Additional reward for successful landing (both legs in contact)
        landing_reward = 0
        if leg1 and leg2:
            landing_reward = self.landing_bonus * (1.0 - min(1.0, abs(x_vel) + abs(y_vel)))
        
        # Add shaped rewards to the original reward
        shaped_reward = reward - velocity_penalty + stability_reward + landing_reward
        
        # Early success detection with higher threshold
        if reward > SUCCESS_REWARD_THRESHOLD:
            shaped_reward += 20  # Bonus for successful landing
            terminated = True
        
        return obs, shaped_reward, terminated, truncated, info

# Enhanced DQN network with Dueling architecture
class DuelingDQN(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DuelingDQN, self).__init__()
        
        # Shared feature layers
        self.features = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_size)
        )
        
    def forward(self, x):
        features = self.features(x)
        value = self.value(features)
        advantage = self.advantage(features)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)

# Enhanced Actor network with better initialization
class EnhancedActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(EnhancedActor, self).__init__()
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Output layer with appropriate initialization for continuous control
        self.mu = nn.Sequential(
            nn.Linear(256, act_size),
            nn.Tanh(),
        )
        
        # Initialize the output layer to produce near-zero outputs initially
        for layer in self.mu:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                nn.init.uniform_(layer.bias, -3e-3, 3e-3)
        
        # Separate trainable standard deviation parameter for each action
        self.logstd = nn.Parameter(torch.zeros(act_size) - 0.5)  # Initialize to smaller values
        
    def forward(self, x):
        x = self.net(x)
        return self.mu(x)

# Enhanced Critic network with layer normalization instead of batch normalization
class EnhancedCritic(nn.Module):
    def __init__(self, obs_size):
        super(EnhancedCritic, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),  # Layer norm works with batch size 1
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),  # Layer norm works with batch size 1
            nn.Linear(256, 1)
        )
        
        # Initialize the output layer with small weights for better stability
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias, -3e-3, 3e-3)
        
    def forward(self, x):
        return self.net(x)

# Prioritized Experience Replay for DQN
class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.buffer = []
        self.position = 0
        self.size = size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.frame = 1
        
    def beta(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, experience):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.size
        self.frame += 1
    
    def sample(self, batch_size):
        if len(self.buffer) == self.size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta())
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

# DQN agent with double Q-learning
class AdvancedDQNAgent(ptan.agent.BaseAgent):
    def __init__(self, net, target_net, device="cpu", epsilon=0.05, discretize_count=DISCRETIZE_COUNT):
        self.net = net
        self.target_net = target_net
        self.device = device
        self.epsilon = epsilon
        
        # Create discretized action space for DQN
        self.actions = []
        # Simple grid discretization (ensures action count matches network output)
        for m1 in np.linspace(-1, 1, discretize_count):
            for m2 in np.linspace(-1, 1, discretize_count):
                self.actions.append([m1, m2])
        self.actions = torch.FloatTensor(self.actions).to(device)
        self.action_count = len(self.actions)
        
        # Debugging information
        print(f"DQN Agent initialized with {self.action_count} discrete actions")
        print(f"Network output size: {net.advantage[2].out_features}")
        assert self.action_count == net.advantage[2].out_features, "Action count mismatch!"
    
    def __call__(self, states, agent_states=None):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        
        # Double DQN action selection - use online net to select action, target net to evaluate it
        q_vals_v = self.net(states_v)
        
        actions = []
        for q_val in q_vals_v:
            if np.random.random() < self.epsilon:
                act_idx = np.random.randint(0, self.action_count)
            else:
                act_idx = q_val.argmax().item()
                
            # Safety check to prevent index out of bounds errors
            if act_idx >= self.action_count:
                print(f"Warning: Action index {act_idx} out of bounds, clipping to {self.action_count-1}")
                act_idx = self.action_count - 1
                
            actions.append(self.actions[act_idx].cpu().numpy())
        
        return actions, agent_states

# A2C agent with advanced features
class AdvancedA2CAgent(ptan.agent.BaseAgent):
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

# Calculate GAE (Generalized Advantage Estimation)
def calc_gae(rewards, values, next_value, dones, gamma, gae_lambda):
    values = np.append(values, next_value)
    gae = 0
    returns = []
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    
    return returns

# Enhanced training function for A2C with GAE and PPO-style clipping
def train_a2c(args, device):
    model_save_path = SAVE_DIR / "a2c"
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    # Create environments with enhanced wrappers
    env_id = "LunarLander-v2"
    # Create environments with enhanced wrappers in a specific order
    envs = []
    for _ in range(ENVS_COUNT):
        env = gym.make(env_id, continuous=True)
        env = StateNormalizer(env)  # First normalize states
        env = RewardShaper(env)     # Then apply reward shaping
        envs.append(env)
    
    test_env = gym.make(env_id, continuous=True)
    test_env = StateNormalizer(test_env)
    test_env = RewardShaper(test_env)
    
    # Get dimensions
    obs_size = envs[0].observation_space.shape[0]
    act_size = envs[0].action_space.shape[0]
    
    # Create advanced networks
    net_act = EnhancedActor(obs_size, act_size).to(device)
    net_crt = EnhancedCritic(obs_size).to(device)
    print(net_act)
    print(net_crt)
    
    # Create tensorboard writer
    writer = SummaryWriter(comment=f"-{args.name}_a2c_enhanced")
    
    # Create agent and experience source
    agent = AdvancedA2CAgent(net_act, device=device, epsilon=EPS_START)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=REWARD_STEPS)
    
    # Create optimizers with learning rate scheduling
    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)
    
    # Learning rate schedulers
    lr_scheduler_act = optim.lr_scheduler.CosineAnnealingLR(opt_act, T_max=100000, eta_min=LEARNING_RATE_ACTOR/10)
    lr_scheduler_crt = optim.lr_scheduler.CosineAnnealingLR(opt_crt, T_max=100000, eta_min=LEARNING_RATE_CRITIC/10)
    
    # Variables for tracking
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
                
                # Process rewards
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    mean_reward = np.mean(rewards)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(mean_reward, step_idx)
                    reward_history.append((step_idx, mean_reward))
                
                # Testing
                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = model.test_net(net_act, test_env, device=device)
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
                            
                            # Record video
                            if rewards > 180:  # Only record good performances
                                params = {"lr_actor": lr_scheduler_act.get_last_lr()[0], 
                                          "lr_critic": lr_scheduler_crt.get_last_lr()[0]}
                                record_video("A2C", net_act, env_id, device, params, rewards, step_idx)
                        
                        best_reward = rewards
                
                # Collect experience
                batch.append(exp)
                
                # Process batch
                if len(batch) < BATCH_SIZE:
                    continue
                
                # Process collected batch for A2C training
                states_v, actions_v, vals_ref_v = common.unpack_batch_a2c(
                    batch, net_crt, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                batch.clear()
                
                # Train critic
                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                # Gradient clipping for critic
                torch.nn.utils.clip_grad_norm_(net_crt.parameters(), max_norm=0.5)
                opt_crt.step()
                
                # Train actor
                opt_act.zero_grad()
                mu_v = net_act(states_v)
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                
                # Normalize advantages
                adv_v = (adv_v - adv_v.mean()) / (adv_v.std() + 1e-8)
                
                # Calculate log probability
                log_prob_v = model.calc_logprob(mu_v, net_act.logstd, actions_v)
                
                # Calculate old log probability for PPO-style clipping
                old_log_prob_v = log_prob_v.detach()
                
                # PPO-style objective with clipping
                ratio_v = torch.exp(log_prob_v - old_log_prob_v)
                surr1_v = adv_v * ratio_v
                surr2_v = adv_v * torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                loss_policy_v = -torch.min(surr1_v, surr2_v).mean()
                
                # Entropy loss for exploration
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
                
                # Total loss
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(net_act.parameters(), max_norm=0.5)
                opt_act.step()
                
                # Update learning rates
                lr_scheduler_act.step()
                lr_scheduler_crt.step()
                
                # Track metrics
                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("lr_actor", lr_scheduler_act.get_last_lr()[0], step_idx)
                tb_tracker.track("lr_critic", lr_scheduler_crt.get_last_lr()[0], step_idx)
    
    return reward_history, epsilon_history

# Enhanced training function for DQN with prioritized replay, double DQN and dueling architecture
def train_dqn(args, device):
    model_save_path = SAVE_DIR / "dqn"
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    # Create environments with enhanced wrappers
    env_id = "LunarLander-v2"
    env = gym.make(env_id, continuous=True)
    env = StateNormalizer(env)
    env = RewardShaper(env)
    
    test_env = gym.make(env_id, continuous=True)
    test_env = StateNormalizer(test_env)
    test_env = RewardShaper(test_env)
    
    # Get dimensions
    obs_size = env.observation_space.shape[0]
    
    # For DQN, we discretize the action space
    # Total number of discrete actions (DISCRETIZE_COUNT^2)
    action_size = DISCRETIZE_COUNT ** 2
    
    # Create dueling DQN networks
    net = DuelingDQN(obs_size, action_size).to(device)
    tgt_net = DuelingDQN(obs_size, action_size).to(device)
    tgt_net.load_state_dict(net.state_dict())
    
    print(net)
    
    # Create tensorboard writer
    writer = SummaryWriter(comment=f"-{args.name}_dqn_enhanced")
    
    # Create agent and experience source
    agent = AdvancedDQNAgent(net, tgt_net, device=device, epsilon=EPS_START)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=N_STEPS)
    
    # Use prioritized replay buffer
    buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
    
    # Create optimizer with learning rate scheduler
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE_DQN)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=LEARNING_RATE_DQN/10)
    
    # Variables for tracking
    reward_history = []
    epsilon_history = []
    best_reward = None
    
    # Frame buffer for n-step returns
    frame_buffer = deque(maxlen=N_STEPS)
    
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx in range(1, 1000000):
                # Collect experience
                frame_buffer.clear()
                
                for _ in range(N_STEPS):
                    # Get a new experience
                    exp = next(iter(exp_source))
                    frame_buffer.append(exp)
                    
                    # Add to replay buffer
                    buffer.push(exp)
                
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
                
                # Testing
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
                            
                            # Record video
                            if rewards > 180:  # Only record good performances
                                params = {"lr": lr_scheduler.get_last_lr()[0]}
                                record_video("DQN", net, env_id, device, params, rewards, step_idx)
                        
                        best_reward = rewards
                
                # Sync target network
                if step_idx % TARGET_NET_SYNC == 0:
                    tgt_net.load_state_dict(net.state_dict())
                
                # Wait until enough samples in buffer
                if len(buffer) < REPLAY_INITIAL:
                    continue
                
                # Sample batch with priorities
                batch, indices, weights = buffer.sample(BATCH_SIZE)
                
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
                weights_v = torch.FloatTensor(weights).to(device)
                
                # Train Double DQN
                optimizer.zero_grad()
                
                # Current Q-values
                q_vals_v = net(states_v)
                state_action_values = q_vals_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
                
                # Next state values using Double Q-learning
                next_state_actions = net(next_states_v).max(1)[1]
                next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
                next_state_values[done_mask] = 0.0
                
                # Expected Q-values
                expected_state_action_values = rewards_v + (GAMMA ** N_STEPS) * next_state_values.detach()
                
                # Compute loss with importance sampling weights for prioritized replay
                losses_v = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
                weighted_losses = losses_v * weights_v
                loss_v = weighted_losses.mean()
                
                # Backpropagation
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update learning rate
                lr_scheduler.step()
                
                # Update priorities in buffer
                td_errors = (state_action_values - expected_state_action_values).abs().data.cpu().numpy()
                buffer.update_priorities(indices, td_errors + 1e-5)  # Add small constant to avoid zero priorities
                
                # Track metrics
                tb_tracker.track("loss", loss_v, step_idx)
                tb_tracker.track("td_error", td_errors.mean(), step_idx)
                tb_tracker.track("lr", lr_scheduler.get_last_lr()[0], step_idx)
                
                if step_idx % 10000 == 0:
                    print(f"Processed {step_idx} steps, buffer size={len(buffer)}")
    
    return reward_history, epsilon_history

# Calculate epsilon based on current step (exponential decay)
def get_epsilon(step):
    return EPS_FINAL + (EPS_START - EPS_FINAL) * math.exp(-step / EPS_DECAY)

# Function to record video
def record_video(model_type, model, env_id, device, params, reward, step_idx):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create recording environment
    env = gym.make(env_id, continuous=True, render_mode="rgb_array")
    env = StateNormalizer(env)  # Use state normalization for consistent behavior
    env = RecordVideo(
        env, 
        video_folder=str(VIDEO_DIR),
        episode_trigger=lambda e: True,
        name_prefix=f"{model_type}_step{step_idx}"
    )
    
    # Run episode with the trained model
    if model_type == "A2C":
        agent = AdvancedA2CAgent(model, device=device, epsilon=0.0)
    else:  # DQN
        target_net = DuelingDQN(env.observation_space.shape[0], DISCRETIZE_COUNT**2).to(device)
        target_net.load_state_dict(model.state_dict())
        agent = AdvancedDQNAgent(model, target_net, device=device, epsilon=0.0)
    
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
    print(f"Video created at: {latest_video}")
    
    # Create a simple text file with the parameters
    info_text = (f"Model: {model_type}\n"
                f"Reward: {reward:.2f}\n"
                f"Step: {step_idx}\n"
                f"Parameters: {params}\n")
    
    info_file = VIDEO_DIR / f"{model_type}_info_{timestamp}.txt"
    with open(info_file, 'w') as f:
        f.write(info_text)
    
    print(f"Model information saved to: {info_file}")
    print(f"Video parameters: {info_text}")
    
    return latest_video

# Function to test DQN model
def test_dqn(net, env, device="cpu", episodes=5):
    rewards = 0.0
    steps = 0
    # Create a target net copy for the agent
    target_net = DuelingDQN(env.observation_space.shape[0], DISCRETIZE_COUNT**2).to(device)
    target_net.load_state_dict(net.state_dict())
    agent = AdvancedDQNAgent(net, target_net, device=device, epsilon=0.0)  # No exploration during testing
    
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

# Create comparison plots
def create_comparison_plots(a2c_data, dqn_data, eps_data):
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
    plt.savefig(PLOTS_DIR / 'reward_comparison.png')
    
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
    plt.savefig(PLOTS_DIR / 'moving_avg_reward.png')

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
    
    # Create directories
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR = Path("plots")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train models based on selection
    if args.model in ["a2c", "both"]:
        print("Training A2C model with advanced techniques...")
        a2c_rewards, a2c_eps = train_a2c(args, device)
        a2c_data = a2c_rewards
        if not eps_data:
            eps_data = a2c_eps
    
    if args.model in ["dqn", "both"]:
        print("Training DQN model with advanced techniques...")
        dqn_rewards, dqn_eps = train_dqn(args, device)
        dqn_data = dqn_rewards
        if not eps_data:
            eps_data = dqn_eps
    
    # Create comparison plots
    create_comparison_plots(a2c_data, dqn_data, eps_data)
    
    print("Training complete!")
    print("Check the 'videos' directory for recordings of successful landings")
    print("Check the 'plots' directory for performance comparison plots")