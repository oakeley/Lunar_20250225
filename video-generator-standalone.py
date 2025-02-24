#!/usr/bin/env python3
import os
import sys
import argparse
import time
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Ensure we can import from the current directory
sys.path.append('.')

# Constants
VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)

def generate_video(model_path, model_type, parameters, episodes=1):
    """Generate a video of the lunar lander using the specified model"""
    try:
        # Import required modules
        from lib import model
        
        # Set up environment
        env_id = "LunarLander-v2"
        env = gym.make(env_id, continuous=True, render_mode="rgb_array")
        
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create a timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Configure recording
        record_env = RecordVideo(
            env,
            video_folder=str(VIDEO_DIR),
            episode_trigger=lambda episode_id: True,
            name_prefix=f"{model_type}_landing_{timestamp}"
        )
        
        # Load model
        if model_type.lower() == "a2c":
            # Get observation and action dimensions
            obs_size = env.observation_space.shape[0]
            act_size = env.action_space.shape[0]
            net = model.ModelActor(obs_size, act_size).to(device)
            net.load_state_dict(torch.load(model_path, map_location=device))
            
            # Run episodes
            for episode in range(episodes):
                obs, _ = record_env.reset()
                done = False
                total_reward = 0
                steps = 0
                
                while not done:
                    # Get action from model
                    obs_v = torch.FloatTensor([obs]).to(device)
                    mu_v = net(obs_v)
                    action = mu_v.squeeze(dim=0).data.cpu().numpy()
                    action = np.clip(action, -1, 1)
                    
                    # Take action
                    obs, reward, terminated, truncated, _ = record_env.step(action)
                    total_reward += reward
                    steps += 1
                    done = terminated or truncated
                
                print(f"Episode {episode+1} finished with reward {total_reward:.2f} in {steps} steps")
        
        else:
            print(f"Model type {model_type} not supported yet")
            return None
        
        record_env.close()
        
        # Find the most recent video file
        video_files = sorted(VIDEO_DIR.glob("*.mp4"), key=os.path.getmtime)
        if not video_files:
            print("No video file was created!")
            return None
        
        latest_video = video_files[-1]
        
        # Create a text file with information about the run
        info_text = (
            f"Model Type: {model_type}\n"
            f"Model Path: {model_path}\n"
            f"Parameters:\n"
        )
        
        for key, value in parameters.items():
            info_text += f"  {key}: {value}\n"
            
        info_text += f"Final Reward: {total_reward:.2f}\n"
        info_text += f"Steps: {steps}\n"
        info_text += f"Device: {device}\n"
        
        info_file = VIDEO_DIR / f"{model_type}_info_{timestamp}.txt"
        with open(info_file, 'w') as f:
            f.write(info_text)
        
        print(f"Video saved to: {latest_video}")
        print(f"Info saved to: {info_file}")
        
        return latest_video
        
    except Exception as e:
        print(f"Error generating video: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video of Lunar Lander using a trained model")
    parser.add_argument("--model-path", required=True, help="Path to the model file")
    parser.add_argument("--model-type", choices=["a2c", "dqn"], required=True, help="Type of model (a2c or dqn)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate used in training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor used in training")
    
    args = parser.parse_args()
    
    # Set up parameters
    parameters = {
        "learning_rate": args.lr,
        "gamma": args.gamma
    }
    
    # Generate video
    generate_video(args.model_path, args.model_type, parameters, args.episodes)
