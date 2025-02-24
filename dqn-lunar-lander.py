#!/usr/bin/env python3
import os
import math
import time
import gymnasium as gym
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from collections import deque
import random
from gymnasium.wrappers import RecordVideo

# Force GPU detection
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available, using CPU")

# Simple hyperparameters
GAMMA = 0.99
LEARNING_RATE = 3e-4
BUFFER_SIZE = 100000
BATCH_SIZE = 64
UPDATE_EVERY = 4
TAU = 1e-3  # For soft update of target network
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 200000

# Paths for saving models and videos
SAVE_DIR = Path("saves")
VIDEO_DIR = Path("videos")

# Create specialized reward wrapper for faster learning
class EnhancedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EnhancedRewardWrapper, self).__init__(env)
        self.episode_steps = 0
        self.prev_shaping = None
        
    def reset(self, **kwargs):
        self.episode_steps = 0
        self.prev_shaping = None
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        
        # Extract state information
        x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1, leg2 = obs
        
        # Additional reward shaping
        shaping = (
            - 100 * abs(x_pos)                     # Penalty for being off-center
            - 100 * abs(angle)                     # Penalty for tilting
            - 100 * abs(angular_vel)               # Penalty for rotating
            - 50 * abs(x_vel)                      # Penalty for horizontal speed
            - 50 * abs(y_vel) if y_vel > 0 else 0  # Penalty for upward velocity
        )
        
        # Delta from previous shaping
        if self.prev_shaping is not None:
            reward += shaping - self.prev_shaping
        self.prev_shaping = shaping
        
        # Extra rewards for good states
        if leg1 and leg2:  # Both legs on ground
            reward += 10
            
            # Bonus for perfect landing
            if abs(angle) < 0.1 and abs(angular_vel) < 0.1 and abs(x_vel) < 0.1:
                reward += 50
                # If this wasn't already a terminal state, make it one
                if not terminated:
                    terminated = True
                    reward += 100  # Big bonus for successful landing
        
        # Penalty for going off-screen or spending too much time
        if terminated and reward < 100:  # If terminated for bad reasons
            reward -= 100
            
        return obs, reward, terminated, truncated, info

# Simple network
class SimpleQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super(SimpleQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Simpler architecture with proper initialization
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
        # Initialize with reasonable weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Simple replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed=0):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = deque(maxlen=4)  # For N-step returns
        self.gamma = GAMMA
    
    def add(self, state, action, reward, next_state, done):
        self.experience.append((state, action, reward, next_state, done))
        
        # If we have enough experiences or reached a terminal state
        if len(self.experience) == 4 or done:
            # Calculate n-step return
            state, action, _, _, _ = self.experience[0]
            
            # Compute n-step discounted reward
            n_reward = 0
            for i, (_, _, r, _, d) in enumerate(self.experience):
                if d and i < len(self.experience) - 1:  # If terminated before last step
                    break
                n_reward += (self.gamma ** i) * r
            
            # Get the final next_state
            _, _, _, next_state, done = self.experience[-1]
            
            # Add the processed experience to memory
            e = (state, action, n_reward, next_state, done)
            self.memory.append(e)
            
            # Remove the earliest experience
            self.experience.popleft()
            
            # If done, clear the remaining experiences
            if done:
                self.experience.clear()
    
    def sample(self):
        experiences = random.sample(self.memory, k=min(len(self.memory), self.batch_size))
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([1 if e[4] else 0 for e in experiences]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, device, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.seed = random.seed(seed)
        
        # Q-Networks
        self.qnetwork_local = SimpleQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = SimpleQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        # Prioritize successful landings in replay
        self.success_memory = ReplayBuffer(1000, min(BATCH_SIZE, 16), seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # Steps done for epsilon calculation
        self.steps_done = 0
        
        print(self.qnetwork_local)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Save especially good experiences separately
        if reward > 100:  # If it was a good landing
            self.success_memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
                # If we have successful examples, learn from those too (more frequently)
                if len(self.success_memory) > 16:
                    success_experiences = self.success_memory.sample()
                    self.learn(success_experiences, GAMMA, weight=2.0)  # Higher weight for success

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        self.steps_done += 1
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, weight=1.0):
        """Update value parameters using batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss with optional weight
        loss = F.mse_loss(Q_expected, Q_targets) * weight
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def get_epsilon(self):
        """Calculate current epsilon based on steps done"""
        return max(EPSILON_END, EPSILON_START * 
                   (EPSILON_END/EPSILON_START) ** (self.steps_done / EPSILON_DECAY))
            
# Train the agent
def train_dqn(env, agent, n_episodes=10000, max_steps=1000):
    """Train DQN agent."""
    scores = []
    scores_window = deque(maxlen=100)
    best_score = -float('inf')
    episode_count = 0
    frame_idx = 0
    
    # Training speed tracking
    training_start = time.time()
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        for t in range(max_steps):
            frame_idx += 1
            # Act with epsilon-greedy
            epsilon = agent.get_epsilon()
            action = agent.act(state, eps=epsilon)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
                
        episode_count += 1
        scores.append(score)
        scores_window.append(score)
        
        # Print training progress
        if i_episode % 100 == 0:
            fps = frame_idx / (time.time() - training_start)
            print(f"\rEpisode {i_episode}\tAvg Score: {np.mean(scores_window):.2f}\tEpsilon: {epsilon:.2f}\tFPS: {fps:.2f}")
        
        # Check if we've reached a new performance peak
        if np.mean(scores_window) > best_score:
            best_score = np.mean(scores_window)
            # Save agent's local network
            torch.save(agent.qnetwork_local.state_dict(), f"{SAVE_DIR}/dqn_best.pth")
            
            # Record video if performance is good
            if best_score > 200:
                record_video(agent, env.unwrapped.spec.id, agent.device, best_score, i_episode)
    
    return scores

# Function to record video
def record_video(agent, env_id, device, score, episode):
    """Record a video of the agent's performance"""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create recording environment
    env = gym.make(env_id, render_mode="rgb_array")
    env = EnhancedRewardWrapper(env)
    env = RecordVideo(
        env, 
        video_folder=str(VIDEO_DIR),
        episode_trigger=lambda e: True,
        name_prefix=f"dqn_ep{episode}_score{score:.1f}"
    )
    
    # Run one episode
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state, eps=0)  # No exploration during evaluation
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        done = terminated or truncated
        
    env.close()
    
    print(f"\nVideo recorded with score: {total_reward:.2f}")

# Function for evaluating the agent
def evaluate(agent, env, n_episodes=10):
    """Evaluate the agent on several episodes"""
    scores = []
    for i in range(n_episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.act(state, eps=0.0)  # No exploration during evaluation
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            score += reward
            done = terminated or truncated
            
        scores.append(score)
        print(f"Episode {i+1}: Score = {score:.2f}")
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Avg Score: {np.mean(scores):.2f}")
    print(f"  Min Score: {np.min(scores):.2f}")
    print(f"  Max Score: {np.max(scores):.2f}")
    
    return np.mean(scores)

def create_discrete_env():
    """Create LunarLander environment with discretized action space"""
    # Create base environment
    env = gym.make('LunarLander-v2')  # Use non-continuous version
    env = EnhancedRewardWrapper(env)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="auto", help="Device to use: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Mode to run in")
    args = parser.parse_args()
    
    # Auto-detect CUDA if available
    if args.dev == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.dev)
    
    print(f"Using device: {device}")
    
    # Create directories
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create environment and agent
    env = create_discrete_env()
    
    # Get state and action space sizes
    state_size = env.observation_space.shape[0]  # 8 features
    action_size = env.action_space.n  # 4 actions
    
    agent = DQNAgent(state_size, action_size, device)
    
    if args.mode == "train":
        # Train the agent
        print("Training DQN agent with enhanced reward...")
        scores = train_dqn(env, agent)
        print("Training complete!")
    else:
        # Load the trained model
        agent.qnetwork_local.load_state_dict(torch.load(f"{SAVE_DIR}/dqn_best.pth", map_location=device))
        # Evaluate and record video
        print("Evaluating agent...")
        score = evaluate(agent, env)
        record_video(agent, env.unwrapped.spec.id, device, score, 0)
    
    env.close()

if __name__ == "__main__":
    main()
