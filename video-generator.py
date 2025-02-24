#!/usr/bin/env python3
import os
import os
import sys
import argparse
import time
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path
import numpy as np
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ImageClip
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2

# Import model classes from the training script
sys.path.append('.')  # Ensure we can import from the current directory
from lib import model
from lunar_lander_enhanced import ModelDQN, AgentDQN, AgentA2C

# Constants
VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)

def create_info_frame(model_type, parameters, reward, episode, width=640, height=120):
    """Create an information frame to append to the video"""
    # Create a blank image
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font if available, otherwise use default
    try:
        font = ImageFont.truetype("Arial", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((20, 10), f"Model: {model_type}", fill=(255, 255, 255), font=font)
    draw.text((20, 30), f"Reward: {reward:.2f}", fill=(255, 255, 255), font=font)
    draw.text((20, 50), f"Episode: {episode}", fill=(255, 255, 255), font=font)
    
    # Add parameters on the right side
    y_offset = 10
    for key, value in parameters.items():
        draw.text((320, y_offset), f"{key}: {value}", fill=(255, 255, 255), font=font)
        y_offset += 20
    
    # Convert to numpy array for moviepy
    return np.array(img)

def generate_video(model_path, model_type, parameters, save_path=None):
    """Generate a video of the lunar lander using the specified model"""
    # Set up environment
    env_id = "LunarLander-v2"
    env = gym.make(env_id, continuous=True, render_mode="rgb_array")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model based on type
    if model_type.lower() == "a2c":
        # Get observation and action dimensions
        obs_size = env.observation_space.shape[0]
        act_size = env.action_space.shape[0]
        net = model.ModelActor(obs_size, act_size).to(device)
        net.load_state_dict(torch.load(model_path, map_location=device))
        agent = AgentA2C(net, device=device, epsilon=0.0)  # No exploration for video
    elif model_type.lower() == "dqn":
        # Get observation dimension
        obs_size = env.observation_space.shape[0]
        # DQN uses discretized action space
        discretize_count = parameters.get("discretize_count", 5)
        action_size = discretize_count ** 2
        net = ModelDQN(obs_size, action_size).to(device)
        net.load_state_dict(torch.load(model_path, map_location=device))
        agent = AgentDQN(net, device=device, epsilon=0.0, discretize_count=discretize_count)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create unique video filename
    if save_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = VIDEO_DIR / f"{model_type}_landing_{timestamp}.mp4"
    
    # Set up recording
    env = RecordVideo(
        env,
        video_folder=str(VIDEO_DIR),
        episode_trigger=lambda episode_id: True,
        name_prefix=f"{model_type}_episode"
    )
    
    # Run episode
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    frames = []
    
    while not done:
        # Capture frame
        frame = env.render()
        frames.append(frame)
        
        # Get action from agent
        actions, _ = agent([obs])
        
        # Step environment
        obs, reward, terminated, truncated, _ = env.step(actions[0])
        
        total_reward += reward
        steps += 1
        done = terminated or truncated
    
    env.close()
    
    # Create info frame
    info_params = {
        "Learning Rate": parameters.get("lr", "N/A"),
        "Gamma": parameters.get("gamma", 0.99),
        "Epsilon": parameters.get("epsilon", 0.0),
        "Steps": steps
    }
    
    info_frame = create_info_frame(model_type, info_params, total_reward, 1)
    
    # Check if we have frames
    if not frames:
        print("No frames captured!")
        return None
    
    # Find the most recent video file created by the RecordVideo wrapper
    video_files = sorted(VIDEO_DIR.glob("*.mp4"), key=os.path.getmtime)
    if not video_files:
        print("No video file was created!")
        return None
    
    latest_video = video_files[-1]
    
    # Add overlay with parameters and metrics
    try:
        # Load the video
        video_clip = VideoFileClip(str(latest_video))
        
        # Create information panel as a separate clip
        info_clip = ImageClip(info_frame).set_duration(video_clip.duration)
        
        # Calculate new dimensions to include info panel below the video
        new_width = video_clip.w
        new_height = video_clip.h + info_clip.h
        
        # Position clips
        video_clip = video_clip.set_position((0, 0))
        info_clip = info_clip.set_position((0, video_clip.h))
        
        # Combine clips
        final_clip = CompositeVideoClip([video_clip, info_clip], size=(new_width, new_height))
        
        # Write the final video
        final_clip.write_videofile(str(save_path), codec='libx264', fps=30)
        
        print(f"Video saved to {save_path}")
        return save_path
    
    except Exception as e:
        print(f"Error creating video: {e}")
        return latest_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video of Lunar Lander using a trained model")
    parser.add_argument("--model-path", required=True, help="Path to the model file")
    parser.add_argument("--model-type", choices=["a2c", "dqn"], required=True, help="Type of model (a2c or dqn)")
    parser.add_argument("--output", help="Output video file path")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate used in training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor used in training")
    parser.add_argument("--discretize", type=int, default=5, help="Discretization count for DQN")
    
    args = parser.parse_args()
    
    # Set up parameters
    parameters = {
        "lr": args.lr,
        "gamma": args.gamma,
        "discretize_count": args.discretize
    }
    
    # Generate video
    generate_video(args.model_path, args.model_type, parameters, args.output)
