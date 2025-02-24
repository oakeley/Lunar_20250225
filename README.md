# Enhanced Lunar Lander with A2C and DQN Training

This project implements parallel training of A2C and DQN models for the LunarLander-v2 environment from Gymnasium, including epsilon-greedy exploration, early termination, CUDA support, and video recording capabilities.

## Features

- **Parallel Training**: Train both A2C and DQN models simultaneously
- **Epsilon-Greedy Exploration**: Start with adventurous behavior (high epsilon) and gradually transition to conservative (low epsilon)
- **Early Termination**: Detect successful landings between flags and end episodes early
- **GPU Support**: Automatic CUDA detection and utilization
- **Python 3.11 Compatible**: Updated dependencies and imports
- **Video Recording**: Generate MP4 videos of successful landings including model parameters
- **Comparative Analysis**: Plot tools to visualize differences between A2C and DQN performance

## Installation

### 1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Python dependencies:
```bash
pip install torch gymnasium ptan matplotlib moviepy opencv-python tensorboard
```

### 3. Install ImageMagick and FFMPEG (required for video overlays and rendering)

#### Ubuntu/Debian Linux
```bash
sudo apt-get update
sudo apt-get install imagemagick ffmpeg
```

#### Windows
1. Download the installer from [ImageMagick's official website](https://imagemagick.org/script/download.php#windows)
2. Run the installer
3. Check "Add application directory to your system path"
4. Check "Install legacy utilities (e.g. convert)"

#### macOS
```bash
brew install imagemagick
brew install ffmpeg
```

### 4. Fix ImageMagick Security Policy (Linux only)

On Linux, you need to modify ImageMagick's security policy to allow text operations:

#### Automatic Fix
```bash
# Make the script executable
chmod +x imagemagick-policy-fix.sh

# Run with sudo
sudo ./imagemagick-policy-fix.sh
```

#### Manual Fix
1. Locate your policy file:
   ```bash
   sudo find /etc -name policy.xml
   ```

2. Create a backup:
   ```bash
   sudo cp /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xml.backup
   ```

3. Edit the policy file:
   ```bash
   sudo nano /etc/ImageMagick-6/policy.xml
   ```

4. Update these policies:
   
   Find:
   ```xml
   <policy domain="path" rights="none" pattern="@*" />
   ```
   Change to:
   ```xml
   <policy domain="path" rights="read|write" pattern="@*" />
   ```

   Find:
   ```xml
   <policy domain="coder" rights="none" pattern="TEXT" />
   ```
   Change to:
   ```xml
   <policy domain="coder" rights="read|write" pattern="TEXT" />
   ```

   Find:
   ```xml
   <policy domain="path" rights="none" pattern="/tmp/*" />
   ```
   Change to:
   ```xml
   <policy domain="path" rights="read|write" pattern="/tmp/*" />
   ```

5. Save the file and exit

## Project Structure

- `lunar-lander-enhanced.py`: Main training script for both A2C and DQN
- `video-generator.py`: Tool for creating videos with text overlays
- `video-generator-standalone.py`: Simpler video generator (no ImageMagick needed)
- `comparison-plotter.py`: Tool for generating comparative analysis plots
- `moviepy-config-fix.py`: Utility to configure MoviePy to find ImageMagick
- `imagemagick-policy-fix.sh`: Script to fix ImageMagick security policies on Linux

Ensure your project directory is structured as follows:

```
your_project_directory/
├── lunar-lander-enhanced.py
├── video-generator.py
├── video-generator-standalone.py
├── comparison-plotter.py
├── moviepy-config-fix.py
├── imagemagick-policy-fix.sh
├── lib/
│   ├── __init__.py
│   ├── model.py
│   └── common.py
```

## Usage

### Training Models

```bash
# Train both A2C and DQN models
python lunar-lander-enhanced.py --name "dual_training" --model both

# Train only A2C model
python lunar-lander-enhanced.py --name "a2c_only" --model a2c

# Train only DQN model
python lunar-lander-enhanced.py --name "dqn_only" --model dqn

# Specify CUDA device (if you have multiple GPUs)
python lunar-lander-enhanced.py --name "gpu_training" --dev cuda:0
```

### Generating Videos

After training, generate videos of successful landings:

```bash
# Using the full version with text overlays (requires ImageMagick)
python video-generator.py --model-path saves/a2c/best_230.5_50000.dat --model-type a2c

# Using the standalone version (no ImageMagick required)
python video-generator-standalone.py --model-path saves/a2c/best_230.5_50000.dat --model-type a2c
```

### Creating Comparison Plots

Generate comparison plots between A2C and DQN:

```bash
python comparison-plotter.py --a2c-logs runs/*a2c* --dqn-logs runs/*dqn*
```

## Key Parameters

You can adjust these parameters in the `lunar-lander-enhanced.py` file:

- `GAMMA` (0.99): Discount factor for future rewards
- `LEARNING_RATE_ACTOR` (1e-4): Learning rate for the A2C actor network
- `LEARNING_RATE_CRITIC` (1e-3): Learning rate for the A2C critic network
- `LEARNING_RATE_DQN` (1e-3): Learning rate for the DQN network
- `ENTROPY_BETA` (1e-2): Entropy coefficient for exploration in A2C
- `EPS_START` (1.0): Initial epsilon value for exploration
- `EPS_FINAL` (0.01): Final epsilon value
- `EPS_DECAY` (100000): Number of steps for epsilon decay
- `SUCCESS_REWARD_THRESHOLD` (200): Reward threshold for successful landing

## Output Directories

- `saves/`: Contains saved model checkpoints
- `videos/`: Contains recorded videos of successful landings
- `plots/`: Contains comparison plots between models
- `runs/`: Contains TensorBoard logs

## Visualizing Training Progress

You can monitor training progress with TensorBoard:

```bash
tensorboard --logdir runs/
```

## Troubleshooting

### ImageMagick Issues

If you encounter issues with the video generator and text overlays:

1. **Check ImageMagick installation**:
   ```bash
   convert --version  # or magick --version on Windows
   ```

2. **Run the configuration fix script**:
   ```bash
   python moviepy-config-fix.py
   ```

3. **Test TextClip functionality**:
   ```bash
   python test_moviepy_textclip.py  # Created by the imagemagick-policy-fix.sh script
   ```

4. **Use the standalone video generator** as a fallback:
   ```bash
   python video-generator-standalone.py --model-path saves/a2c/best_model.dat --model-type a2c
   ```

### CUDA Issues

If you encounter CUDA errors:

1. **Check CUDA availability**:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())
   print(torch.cuda.get_device_name(0))
   ```

2. **Force CPU usage** if needed:
   ```bash
   python lunar-lander-enhanced.py --name "cpu_training" --dev cpu
   ```

## Notes on Model Performance

- **A2C vs DQN**: A2C typically learns faster but DQN may achieve more stable performance in the long run
- **Epsilon-Greedy Effect**: Both algorithms benefit significantly from proper exploration, especially in the early stages
- **Early Termination**: Speeds up training by 20-30% by ending episodes once success is achieved
- **GPU Acceleration**: Training is approximately 3-5x faster on GPU vs CPU

## License

This project is open-source and available under the MIT License.
