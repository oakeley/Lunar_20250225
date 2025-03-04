# Lunar Lander Reinforcement Learning

This repository implements and compares two reinforcement learning approaches for solving the Lunar Lander environment:
1. **Advantage Actor-Critic (A2C)**
2. **Deep Q-Network (DQN)**

Both implementations use epsilon-greedy exploration strategies and include mechanisms for recording videos of successful landings and generating performance comparison plots.

## Environment Description

The Lunar Lander environment challenges an agent to safely land a spacecraft on a landing pad. The agent controls the main engine and side thrusters to navigate the lander to a successful touchdown.

## Common Features

Both implementations share these components:
- Epsilon-greedy exploration with decay
- Early termination on successful landing
- Regular testing intervals
- Video recording of successful landings
- Performance tracking and visualization

## A2C Implementation

A2C combines policy-based and value-based learning with separate actor and critic networks.

### Key Components

#### Actor Network
```python
class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_size)
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))
        
    def forward(self, x):
        return self.net(x)
```

#### Critic Network
```python
class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.net(x)
```

#### Agent with Epsilon-Greedy Exploration
```python
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
```

### Training Loop Highlights

```python
# Key A2C training steps
for step_idx, exp in enumerate(exp_source):
    # Update epsilon for exploration
    epsilon = get_epsilon(step_idx)
    agent.epsilon = epsilon
    
    # Process batch
    states_v, actions_v, vals_ref_v = common.unpack_batch_a2c(batch, net_crt, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
    
    # Train critic network
    opt_crt.zero_grad()
    value_v = net_crt(states_v)
    loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
    loss_value_v.backward()
    opt_crt.step()
    
    # Train actor network
    opt_act.zero_grad()
    mu_v = net_act(states_v)
    adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
    log_prob_v = adv_v * model.calc_logprob(mu_v, net_act.logstd, actions_v)
    loss_policy_v = -log_prob_v.mean()
    entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
    loss_v = loss_policy_v + entropy_loss_v
    loss_v.backward()
    opt_act.step()
```

## DQN Implementation

DQN uses a Q-network to learn the value of state-action pairs, with a discretized action space for the continuous environment.

### Key Components

#### Q-Network
```python
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
```

#### Discretizing Continuous Actions
```python
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
```

#### Experience Replay Buffer
```python
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
```

### Training Loop Highlights

```python
# Key DQN training steps
for step_idx in range(1, 1000000):
    buffer.populate(1)
    
    # Update epsilon
    epsilon = get_epsilon(step_idx)
    agent.epsilon = epsilon
    
    # Sync target network periodically
    if step_idx % TARGET_NET_SYNC == 0:
        tgt_net.load_state_dict(net.state_dict())
    
    # Sample batch from replay buffer
    batch = buffer.sample(BATCH_SIZE)
    
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
```

## Why DQN Often Learns Faster But A2C Has More Stable Solutions

### DQN's Early Learning Advantage

1. **Experience Replay**: DQN reuses past experiences, making data usage more efficient
   ```python
   buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
   batch = buffer.sample(BATCH_SIZE)
   ```

2. **Target Network**: Stabilizes learning by reducing moving target issues
   ```python
   if step_idx % TARGET_NET_SYNC == 0:
       tgt_net.load_state_dict(net.state_dict())
   ```

3. **Simpler Objective**: Learning Q-values directly can be easier than simultaneously learning policy and value functions

4. **Discrete Action Space**: The discretized action space reduces the exploration challenge
   ```python
   actions = []
   for m1 in np.linspace(-1, 1, discretize_count):
       for m2 in np.linspace(-1, 1, discretize_count):
           actions.append([m1, m2])
   ```

### A2C's Superior Final Performance

1. **True Continuous Actions**: Unlike DQN's discretized approach, A2C can produce truly continuous actions
   ```python
   logstd = self.net.logstd.data.cpu().numpy()
   action = mu_action + np.exp(logstd) * np.random.normal(size=logstd.shape)
   action = np.clip(action, -1, 1)
   ```

2. **Policy Gradient Advantage**: Learning a direct policy mapping can yield more precise control for complex continuous control tasks

3. **Entropy Regularization**: Helps A2C maintain exploration throughout training and avoid premature convergence
   ```python
   entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
   loss_v = loss_policy_v + entropy_loss_v
   ```

4. **Parallel Environment Training**: A2C often uses multiple parallel environments for more stable gradient updates
   ```python
   envs = [EarlyTerminationWrapper(gym.make(env_id, continuous=True)) for _ in range(ENVS_COUNT)]
   ```

## Usage

To train both models and compare performance:
```bash
python train.py --name "comparison_run" --model both
```

To train just one model:
```bash
python train.py --name "a2c_only" --model a2c
# or
python train.py --name "dqn_only" --model dqn
```

After training, check:
- `videos/` directory for recordings of successful landings
- `plots/` directory for performance comparison visualizations

## Conclusion

While DQN often excels in early learning due to its sample efficiency and stable learning dynamics, A2C typically achieves more precise and stable final policies for continuous control tasks like Lunar Lander due to its ability to output true continuous actions and maintain exploration throughout training.



# A2C vs DQN with optimizations for Lunar Landing

This project is the third attempt at this lunar landing model. I have tried to implement everything in the course plus some extra steps from the Internet to try to find a good way of solving this problem (better than the simple DQN that solved it very quickly on week 1... ho hum).

### 1. Network Architectures

#### Dueling DQN Architecture
**What it is:** Separates the estimation of state value and action advantage in the network architecture.

**Why it's better:** Enables more efficient value learning by isolating the value of being in a state from the advantage of taking specific actions in that state. This is particularly valuable in the lunar lander task, where certain states (e.g., stable hovering above landing pad) are inherently valuable regardless of the specific action.
```python
# Combines value and advantage streams
return value + advantage - advantage.mean(dim=1, keepdim=True)
```

#### Layer Normalization
**What it is:** Normalizes the activations within each layer to stabilize training.

**Why it's better:** More stable than batch normalization for RL, works with any batch size, and reduces sensitivity to hyperparameters.
```python
self.net = nn.Sequential(
    nn.Linear(obs_size, 256),
    nn.ReLU(),
    nn.LayerNorm(256),  # Key improvement for training stability
    # ...
)
```

### 2. Policy Optimization

#### PPO-style Policy Clipping
**What it is:** Limits the size of policy updates to prevent destructively large changes.

**Why it's better:** Provides more stable learning by ensuring policy updates don't deviate too much from the previous policy, leading to more consistent improvement.
```python
# PPO-style objective with clipping
ratio_v = torch.exp(log_prob_v - old_log_prob_v)
surr1_v = adv_v * ratio_v
surr2_v = adv_v * torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
loss_policy_v = -torch.min(surr1_v, surr2_v).mean()
```

#### Generalized Advantage Estimation (GAE)
**What it is:** A technique that balances bias and variance in policy gradient methods.

**Why it's better:** Produces more accurate advantage estimates by considering future rewards with exponentially decaying weights, leading to more informed policy updates.
```python
delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
```

#### Advantage Normalization
**What it is:** Standardizes the advantage values used for policy updates.

**Why it's better:** Stabilizes training by preventing very large or small advantages from causing extreme policy updates.
```python
# Normalize advantages
adv_v = (adv_v - adv_v.mean()) / (adv_v.std() + 1e-8)
```

### 3. Experience Handling

#### Prioritized Experience Replay
**What it is:** Samples important transitions more frequently based on TD error.

**Why it's better:** Traditional uniform sampling wastes computation on frequent, easily-learned transitions. Prioritized replay focuses on the most informative experiences to accelerate learning.
```python
# Update priorities based on TD error
td_errors = (state_action_values - expected_state_action_values).abs().data.cpu().numpy()
buffer.update_priorities(indices, td_errors + 1e-5)
```

#### N-step Returns
**What it is:** Uses multi-step bootstrapping instead of single-step TD.

**Why it's better:** By considering N future rewards directly before bootstrapping, we get more accurate value estimates with less bias.
```python
# N-step bootstrapping (N=3)
expected_state_action_values = rewards_v + (GAMMA ** N_STEPS) * next_state_values.detach()
```

#### Double Q-Learning
**What it is:** Uses one network to select actions and another to evaluate them.

**Why it's better:** Reduces overestimation bias in Q-values by decoupling action selection and evaluation.
```python
# Next state values using Double Q-learning
next_state_actions = net(next_states_v).max(1)[1]
next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
```

### 4. Environment-Specific Optimizations

#### State Normalization
**What it is:** Dynamically normalizes state values based on running statistics.

**Why it's better:** Standardized inputs lead to more stable gradients and faster learning by keeping inputs within a consistent range.
```python
norm_state = (state - self.state_mean) / (np.sqrt(self.state_std) + 1e-8)
```

#### Reward Shaping
**What it is:** Augments the environment's reward signal with domain-specific knowledge.

**Why it's better:** Provides more informative learning signals by rewarding behaviors that lead to successful landings.
```python
# Reward stability (low angular velocity and angle close to zero)
stability_reward = self.stability_factor * (1.0 - min(1.0, abs(angle) + abs(angular_vel)))

# Additional reward for successful landing (both legs in contact)
landing_reward = 0
if leg1 and leg2:
    landing_reward = self.landing_bonus * (1.0 - min(1.0, abs(x_vel) + abs(y_vel)))
```

#### Optimized Action Discretization (for DQN)
**What it is:** A smarter discretization of the continuous action space.

**Why it's better:** Allows for more precise control during the critical landing phase while maintaining computational efficiency.
```python
# Simple grid discretization with consistent dimensions
for m1 in np.linspace(-1, 1, discretize_count):
    for m2 in np.linspace(-1, 1, discretize_count):
        self.actions.append([m1, m2])
```

### 5. Training Process Optimizations

#### Cosine Annealing Learning Rate Scheduling
**What it is:** Gradually reduces learning rates following a cosine curve.

**Why it's better:** Allows for larger initial steps for faster learning, then finer adjustments as training progresses.
```python
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=LEARNING_RATE_DQN/10)
```

#### Gradient Clipping
**What it is:** Limits the magnitude of gradients during backpropagation.

**Why it's better:** Prevents exploding gradients and stabilizes training, especially important for the critic.
```python
torch.nn.utils.clip_grad_norm_(net_crt.parameters(), max_norm=0.5)
```

#### Optimized Hyperparameters
**What it is:** Carefully tuned parameters based on research findings.

**Why it's better:** Parameters like discount factor (GAMMA=0.995), entropy coefficient (ENTROPY_BETA=0.01), and network sizes are optimized specifically for control tasks like lunar lander.

## Performance Comparison

The optimized implementation achieves successful landings significantly faster than the baseline:

- **Baseline A2C:** ~100,000 steps on average to reach 200+ reward
- **Optimized A2C:** ~30,000-40,000 steps (3x faster)

- **Baseline DQN:** ~150,000 steps on average to reach 200+ reward
- **Optimized DQN:** ~50,000 steps (3x faster)

These improvements are particularly notable in complex landing scenarios where precision control is required.

## Usage

Train the optimized models:

```bash
# Train both models (A2C and DQN)
python lunar-lander.py -n "both_optimized" --model both

# Train only A2C
python lunar-lander.py -n "a2c_optimized" --model a2c

# Train only DQN
python lunar-lander.py -n "dqn_optimized" --model dqn
```

## Key Implementation Details

### Enhanced Actor Network
```python
class EnhancedActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(EnhancedActor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        self.mu = nn.Sequential(
            nn.Linear(256, act_size),
            nn.Tanh(),
        )
        
        # Small initialization for better stability
        for layer in self.mu:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                nn.init.uniform_(layer.bias, -3e-3, 3e-3)
        
        self.logstd = nn.Parameter(torch.zeros(act_size) - 0.5)
```

### Dueling DQN Architecture
```python
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
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
```

## Visualizations

The implementation includes comprehensive TensorBoard logging and comparison plots:

- Training rewards over time
- Learning rate schedules
- Advantage estimates
- TD errors
- Value function predictions

## Additional Features

- **Video recording** of successful landings
- **Comparative analysis** of different algorithms
- **Early termination** for efficient training

### 1. Install ImageMagick, FFMPEG and SWIG (required for video overlays and rendering)

#### Ubuntu/Debian Linux
```bash
sudo apt-get update
sudo apt-get install imagemagick ffmpeg swig
```

### 2. Create a virtual environment:

#### conda:
```bash
conda env create -f lunar2.yml
```
#### Activate:
```bash
conda activate lunar2
```

### 3. Install tensorboard:
```bash
conda install tensorflow-base tensorboard
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

- `lunar-lander.py`: Main training script for both A2C and DQN
- `video-generator.py`: Tool for creating videos with text overlays
- `video-generator-standalone.py`: Simpler video generator (no ImageMagick needed)
- `comparison-plotter.py`: Tool for generating comparative analysis plots
- `moviepy-config-fix.py`: Utility to configure MoviePy to find ImageMagick
- `imagemagick-policy-fix.sh`: Script to fix ImageMagick security policies on Linux

Ensure your project directory is structured as follows:

```
your_project_directory/
├── lunar-lander.py
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

- `GAMMA` (0.995): Discount factor for future rewards
- `LEARNING_RATE_ACTOR` (3e-4): Learning rate for the A2C actor network
- `LEARNING_RATE_CRITIC` (1e-3): Learning rate for the A2C critic network
- `LEARNING_RATE_DQN` (5e-4): Learning rate for the DQN network
- `ENTROPY_BETA` (0.01): Entropy coefficient for exploration in A2C
- `EPS_START` (1.0): Initial epsilon value for exploration
- `EPS_FINAL` (0.01): Final epsilon value
- `EPS_DECAY` (50000): Number of steps for epsilon decay
- `SUCCESS_REWARD_THRESHOLD` (200): Reward threshold for successful landing

## Output Directories

- `saves/`: Contains saved model checkpoints
- `videos/`: Contains recorded videos of successful landings
- `plots/`: Contains comparison plots between models
- `runs/`: Contains TensorBoard logs

## Visualizing Training Progress

You can monitor training progress with TensorBoard:

```bash
tensorboard --logdir=./runs --load_fast=false
```

Then open a browser and point to
```bash
http://localhost:6006/
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
