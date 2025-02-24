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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from lib import model, common

# Force GPU detection
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available, using CPU")

# Carefully tuned hyperparameters that prioritize landing success
GAMMA = 0.99
REWARD_STEPS = 1  # Simpler, more stable TD(1) estimation
BATCH_SIZE = 64   # Smaller batches for more frequent updates
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
ENTROPY_BETA = 1e-3  # Carefully balanced exploration
ENVS_COUNT = 16      # Multiple environments for diverse experience
TEST_ITERS = 1000    # More frequent testing
TEST_EPISODES = 10   # More thorough testing

# Paths for saving models and videos
SAVE_DIR = Path("saves")
VIDEO_DIR = Path("videos")

# Create specialized wrapper for landing success
class LanderSuccessWrapper(gym.Wrapper):
    def __init__(self, env):
        super(LanderSuccessWrapper, self).__init__(env)
        self.episode_steps = 0
        self.episode_reward = 0
        self.successful_landing = False
        
    def reset(self, **kwargs):
        self.episode_steps = 0
        self.episode_reward = 0
        self.successful_landing = False
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.episode_reward += reward
        
        # Extract state information
        x_pos, y_pos, x_vel, y_vel, angle, angular_vel, leg1, leg2 = obs
        
        # Check for successful landing criteria
        landed = leg1 and leg2  # Both legs touching ground
        upright = abs(angle) < 0.1  # Nearly vertical orientation
        stable = abs(angular_vel) < 0.1  # Not rotating
        slow_descent = abs(y_vel) < 0.1  # Slow vertical velocity
        centered = abs(x_pos) < 0.2  # Close to center
        
        # Determine if this is a successful landing
        if landed and upright and stable and slow_descent and centered:
            self.successful_landing = True
            
            if not terminated:  # Only add bonus if not already terminal
                reward += 10.0  # Bonus for clean landing
                terminated = True  # End episode on success
                
        # Extra bonus for landing between flags
        if terminated and reward > 200:
            self.successful_landing = True
        
        # Assemble info dict with diagnostic data
        info = {
            'successful_landing': self.successful_landing,
            'upright': upright,
            'stable': stable,
            'slow_descent': slow_descent,
            'centered': centered,
            'landed': landed,
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward
        }
        
        return obs, reward, terminated, truncated, info

# Enhanced actor with proper action scaling
class FocusedActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(FocusedActor, self).__init__()
        
        # Network architecture optimized for precise control
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_size),
            nn.Tanh()
        )
        
        # Initialize to smaller stdev for more precise initial actions
        self.logstd = nn.Parameter(torch.zeros(act_size) - 1.0)
        
        # Initialize with small weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        return self.net(x)

# Focused critic for value estimation
class FocusedCritic(nn.Module):
    def __init__(self, obs_size):
        super(FocusedCritic, self).__init__()
        
        # Simpler architecture with proper initialization
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize with small weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        return self.net(x)

# A2C agent with controlled exploration
class FocusedA2CAgent(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device
    
    def __call__(self, states, agent_states=None):
        # Fix: Convert list to ndarray first to avoid the warning
        states_np = np.array(states)
        states_v = torch.FloatTensor(states_np).to(self.device)
        
        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        
        actions = []
        for mu_action in mu:
            # Use stochastic policy with correct variance scaling
            logstd = self.net.logstd.data.cpu().numpy()
            
            # Add noise based on trained stdev
            action = mu_action + np.exp(logstd) * np.random.normal(size=logstd.shape)
            
            # Ensure action is within valid range
            action = np.clip(action, -1, 1)
            actions.append(action)
        
        return actions, agent_states

# Fixed batch processing to avoid warnings
def unpack_batch_a2c_fixed(batch, net_crt, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors with proper numpy array conversion
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
            
    # Convert to numpy arrays first to avoid warning
    states_v = torch.FloatTensor(np.array(states)).to(device)
    actions_v = torch.FloatTensor(np.array(actions)).to(device)
    
    # Handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states)).to(device)
        last_vals_v = net_crt(last_states_v)
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v

# Function for detailed testing
def test_net(net, env, count=TEST_EPISODES, device="cpu"):
    rewards = 0.0
    steps = 0
    success_count = 0
    agent = FocusedA2CAgent(net, device=device)
    
    # Metrics to track
    landing_stats = {
        'upright': 0,
        'stable': 0,
        'slow_descent': 0,
        'centered': 0,
        'landed': 0,
        'successful_landing': 0
    }
    
    for _ in range(count):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            actions, _ = agent([obs])
            obs, reward, terminated, truncated, info = env.step(actions[0])
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            
            # If this is the last step, record landing stats
            if done:
                for key in landing_stats.keys():
                    if key in info and info[key]:
                        landing_stats[key] += 1
        
        if info.get('successful_landing', False):
            success_count += 1
            
        rewards += episode_reward
        steps += episode_steps
    
    # Report detailed statistics
    print(f"Test Results ({count} episodes):")
    print(f"  Avg Reward: {rewards/count:.2f}")
    print(f"  Avg Steps: {steps/count:.2f}")
    print(f"  Success Rate: {success_count/count*100:.1f}%")
    print(f"  Landing Stats (% of episodes):")
    for key, value in landing_stats.items():
        print(f"    {key}: {value/count*100:.1f}%")
    
    return rewards / count, steps / count, success_count / count

# Function to record video
def record_video(model_type, model, env_id, device, params, reward, step_idx):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create recording environment
    env = gym.make(env_id, continuous=True, render_mode="rgb_array")
    env = LanderSuccessWrapper(env)
    env = RecordVideo(
        env, 
        video_folder=str(VIDEO_DIR),
        episode_trigger=lambda e: True,
        name_prefix=f"{model_type}_step{step_idx}"
    )
    
    # Run episode with the model
    agent = FocusedA2CAgent(model, device=device)
    
    # Detailed recording for analysis
    trajectory = []
    
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        actions, _ = agent([obs])
        
        # Record the state and action
        trajectory.append({
            'state': obs.copy(),
            'action': actions[0].copy()
        })
        
        obs, reward, terminated, truncated, info = env.step(actions[0])
        total_reward += reward
        done = terminated or truncated
        
        # Record the outcome
        if done:
            trajectory[-1]['reward'] = reward
            trajectory[-1]['done'] = done
            trajectory[-1]['info'] = {k: v for k, v in info.items() if not isinstance(v, (np.ndarray, list))}
    
    env.close()
    
    # Get latest created video file
    video_files = sorted(VIDEO_DIR.glob("*.mp4"), key=os.path.getmtime)
    if not video_files:
        print("No video file created")
        return None
    
    latest_video = video_files[-1]
    print(f"Video created at: {latest_video}")
    
    # Save trajectory for analysis
    trajectory_file = VIDEO_DIR / f"{model_type}_trajectory_{timestamp}.txt"
    with open(trajectory_file, 'w') as f:
        for i, step in enumerate(trajectory):
            f.write(f"Step {i}:\n")
            f.write(f"  State: {step['state']}\n")
            f.write(f"  Action: {step['action']}\n")
            if 'reward' in step:
                f.write(f"  Reward: {step['reward']}\n")
                f.write(f"  Done: {step['done']}\n")
                if 'info' in step:
                    f.write(f"  Info: {step['info']}\n")
            f.write("\n")
    
    # Create a simple text file with the parameters
    info_text = (f"Model: {model_type}\n"
                f"Reward: {reward:.2f}\n"
                f"Step: {step_idx}\n"
                f"Parameters: {params}\n")
    
    info_file = VIDEO_DIR / f"{model_type}_info_{timestamp}.txt"
    with open(info_file, 'w') as f:
        f.write(info_text)
    
    print(f"Model information saved to: {info_file}")
    print(f"Trajectory saved to: {trajectory_file}")
    
    return latest_video

# Optimized A2C training with focus on landing success
def train_a2c(args, device):
    model_save_path = SAVE_DIR / "a2c"
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    # Create environments
    env_id = "LunarLander-v2"
    envs = []
    for _ in range(ENVS_COUNT):
        env = gym.make(env_id, continuous=True)
        env = LanderSuccessWrapper(env)  # Our specialized wrapper
        envs.append(env)
    
    test_env = gym.make(env_id, continuous=True)
    test_env = LanderSuccessWrapper(test_env)
    
    # Get dimensions
    obs_size = envs[0].observation_space.shape[0]
    act_size = envs[0].action_space.shape[0]
    
    # Create networks
    net_act = FocusedActor(obs_size, act_size).to(device)
    net_crt = FocusedCritic(obs_size).to(device)
    print(net_act)
    print(net_crt)
    
    # Create tensorboard writer
    writer = SummaryWriter(comment=f"-{args.name}_a2c_focused")
    
    # Create agent and experience source
    agent = FocusedA2CAgent(net_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, steps_count=REWARD_STEPS)
    
    # Create optimizers with conservative scheduling
    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)
    
    # Simple step-based learning rate decay
    act_scheduler = optim.lr_scheduler.StepLR(opt_act, step_size=100000, gamma=0.5)
    crt_scheduler = optim.lr_scheduler.StepLR(opt_crt, step_size=100000, gamma=0.5)
    
    # Tracking variables
    batch = []
    best_reward = None
    best_success_rate = 0.0
    
    # Training speed tracking
    training_start = time.time()
    frame_idx = 0
    episode_count = 0
    
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                frame_idx += 1
                
                # Process rewards
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    mean_reward = np.mean(rewards)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(mean_reward, step_idx)
                    
                    # Track episode count manually
                    episode_count += len(rewards_steps)
                    
                    # Print training speed every 1000 episodes
                    if step_idx % 1000 == 0:
                        fps = frame_idx / (time.time() - training_start)
                        print(f"{frame_idx}: done {episode_count} episodes, mean reward {mean_reward:.3f}, speed {fps:.2f} f/s")
                
                # Collect experience
                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue
                
                # Prepare batch - use fixed function to avoid warnings
                states_v, actions_v, vals_ref_v = unpack_batch_a2c_fixed(
                    batch, net_crt, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                batch.clear()
                
                # Train critic
                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                torch.nn.utils.clip_grad_norm_(net_crt.parameters(), max_norm=0.1)  # Conservative clipping
                opt_crt.step()
                
                # Train actor
                opt_act.zero_grad()
                mu_v = net_act(states_v)
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = model.calc_logprob(mu_v, net_act.logstd, actions_v)
                loss_policy_v = -log_prob_v.mean()
                
                # Add entropy loss with careful scaling
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
                
                # Combined loss
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(net_act.parameters(), max_norm=0.1)  # Conservative clipping
                opt_act.step()
                
                # Update learning rates
                act_scheduler.step()
                crt_scheduler.step()
                
                # Track metrics
                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("lr_actor", act_scheduler.get_last_lr()[0], step_idx)
                tb_tracker.track("lr_critic", crt_scheduler.get_last_lr()[0], step_idx)
                
                # Periodically test the model with detailed metrics
                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps, success_rate = test_net(net_act, test_env, device=device)
                    print(f"A2C Test done in {time.time() - ts:.2f} sec, reward {rewards:.3f}, steps {steps:.1f}")
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    writer.add_scalar("test_success_rate", success_rate, step_idx)
                    
                    # Save model on improved success rate or reward
                    improved = False
                    
                    # First, check for higher success rate
                    if success_rate > best_success_rate:
                        improved = True
                        best_success_rate = success_rate
                        print(f"Best success rate updated: {best_success_rate*100:.1f}%")
                    
                    # If equal success rate, check for higher reward
                    elif success_rate == best_success_rate and (best_reward is None or rewards > best_reward):
                        improved = True
                        best_reward = rewards
                        print(f"Best reward updated: {best_reward:.3f}")
                    
                    if improved:
                        # Save model
                        name = f"best_sr{best_success_rate:.2f}_r{rewards:.1f}_{step_idx}.dat"
                        fname = model_save_path / name
                        torch.save(net_act.state_dict(), fname)
                        
                        # Record video
                        params = {"lr_actor": act_scheduler.get_last_lr()[0], 
                                  "lr_critic": crt_scheduler.get_last_lr()[0],
                                  "success_rate": success_rate}
                        record_video("A2C", net_act, env_id, device, params, rewards, step_idx)
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="auto", help="Device to use: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
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
    
    # Train A2C
    print("Training A2C model with focus on successful landings...")
    train_a2c(args, device)
    
    print("Training complete!")
    print("Check the 'videos' directory for recordings of successful landings")

if __name__ == "__main__":
    main()