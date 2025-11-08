"""
Training functions for the HRL system.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from tqdm import tqdm
import gc

from .networks import CustomActor, CustomCritic, Layer3ActorCritic
from .memory import ReplayBuffer
from .env import OptionsTradingEnv

def train_layer1(
    obs_df: pd.DataFrame,
    price_data: pd.DataFrame,
    train_dates: List[str],
    all_tickers: List[str],
    seeds: List[int] = [1,2,3,4,5],
    agents_list: List[str] = ["PPO", "TD3", "SAC"],
    num_days: int = 10,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 2000,
    save_dir: str = "./models/layer1"
) -> Dict[str, Any]:
    """
    Train layer 1 RL agents.
    
    Args:
        obs_df: DataFrame with observation vectors
        price_data: DataFrame with price data
        train_dates: List of training dates
        all_tickers: List of ticker symbols
        seeds: List of random seeds
        agents_list: List of agent types to train
        num_days: Option expiry horizon
        learning_rate: Learning rate for optimizers
        batch_size: Batch size for training
        n_epochs: Number of training epochs
        save_dir: Directory to save trained models
        
    Returns:
        Dictionary of trained models
    """
    os.makedirs(save_dir, exist_ok=True)
    trained_models = {}

    for seed in seeds:
        print(f"\n==== Training Seed {seed} ====")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create environment
        env = OptionsTradingEnv(
            all_tickers=all_tickers,
            obs_df=obs_df,
            price_data=price_data,
            train_dates=train_dates,
            num_days=num_days
        )
        
        # Create replay buffer
        buffer = ReplayBuffer(max_size=50000)
        
        for agent_type in agents_list:
            print(f"\nTraining {agent_type}")
            
            # Create networks
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            actor = CustomActor(obs_dim, action_dim)
            critic = CustomCritic(obs_dim, action_dim)
            
            actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
            critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
            
            # Training loop
            episode_rewards = []
            for epoch in tqdm(range(n_epochs), desc=f"{agent_type} Training"):
                obs = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    # Get action
                    with torch.no_grad():
                        action, _ = actor.get_action(torch.FloatTensor(obs).unsqueeze(0))
                        action = action.squeeze(0).numpy()
                    
                    # Take step
                    next_obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                    # Store experience
                    buffer.add(obs, action, reward, next_obs, done)
                    obs = next_obs
                    
                    # Train if enough samples
                    if len(buffer) >= batch_size:
                        experiences = buffer.sample(batch_size)
                        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = experiences
                        
                        # Update critic
                        with torch.no_grad():
                            next_action, _ = actor.get_action(next_obs_batch)
                            target_Q = reward_batch + (1 - done_batch) * 0.99 * critic(next_obs_batch, next_action)
                        
                        current_Q = critic(obs_batch, action_batch)
                        critic_loss = nn.MSELoss()(current_Q, target_Q)
                        
                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        critic_optimizer.step()
                        
                        # Update actor
                        actor_action, _ = actor.get_action(obs_batch)
                        actor_loss = -critic(obs_batch, actor_action).mean()
                        
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()
                
                episode_rewards.append(episode_reward)
                
                if (epoch + 1) % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    print(f"Epoch {epoch+1}, Average Reward: {avg_reward:.2f}")
            
            # Save models
            model_dir = os.path.join(save_dir, f"seed_{seed}")
            os.makedirs(model_dir, exist_ok=True)
            
            torch.save(actor.state_dict(), os.path.join(model_dir, f"{agent_type}_actor.pt"))
            torch.save(critic.state_dict(), os.path.join(model_dir, f"{agent_type}_critic.pt"))
            
            trained_models[(seed, agent_type)] = {
                'actor': actor,
                'critic': critic
            }
            
            # Cleanup
            del actor, critic, actor_optimizer, critic_optimizer
            gc.collect()
            torch.cuda.empty_cache()
        
        # Cleanup environment
        del env
        gc.collect()
    
    return trained_models

def train_layer2(
    obs_df: pd.DataFrame,
    price_data: pd.DataFrame,
    train_dates: List[str],
    all_tickers: List[str],
    seeds: List[int] = [1,2,3,4,5],
    num_days: int = 10,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 2000,
    save_dir: str = "./models/layer2"
) -> Dict[str, Any]:
    """
    Train layer 2 RL agents.
    Similar to layer 1 but with different observation space.
    """
    os.makedirs(save_dir, exist_ok=True)
    trained_models = {}
    
    for seed in seeds:
        print(f"\n==== Training Layer 2 Seed {seed} ====")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        env = OptionsTradingEnv(
            all_tickers=all_tickers,
            obs_df=obs_df,
            price_data=price_data,
            train_dates=train_dates,
            num_days=num_days
        )
        
        buffer = ReplayBuffer(max_size=50000)
        
        # Create networks with larger capacity
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        actor = CustomActor(obs_dim, action_dim, hidden_dims=[1024, 512, 256])
        critic = CustomCritic(obs_dim, action_dim, hidden_dims=[1024, 512, 256])
        
        actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
        critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
        
        # Training loop (similar to layer 1)
        episode_rewards = []
        for epoch in tqdm(range(n_epochs), desc=f"Layer 2 Training"):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    action, _ = actor.get_action(torch.FloatTensor(obs).unsqueeze(0))
                    action = action.squeeze(0).numpy()
                
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs
                
                if len(buffer) >= batch_size:
                    experiences = buffer.sample(batch_size)
                    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = experiences
                    
                    # Update critic
                    with torch.no_grad():
                        next_action, _ = actor.get_action(next_obs_batch)
                        target_Q = reward_batch + (1 - done_batch) * 0.99 * critic(next_obs_batch, next_action)
                    
                    current_Q = critic(obs_batch, action_batch)
                    critic_loss = nn.MSELoss()(current_Q, target_Q)
                    
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()
                    
                    # Update actor
                    actor_action, _ = actor.get_action(obs_batch)
                    actor_loss = -critic(obs_batch, actor_action).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
            
            episode_rewards.append(episode_reward)
            
            if (epoch + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Epoch {epoch+1}, Average Reward: {avg_reward:.2f}")
        
        # Save models
        model_dir = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save(actor.state_dict(), os.path.join(model_dir, "actor.pt"))
        torch.save(critic.state_dict(), os.path.join(model_dir, "critic.pt"))
        
        trained_models[seed] = {
            'actor': actor,
            'critic': critic
        }
        
        # Cleanup
        del actor, critic, actor_optimizer, critic_optimizer, env
        gc.collect()
        torch.cuda.empty_cache()
    
    return trained_models

def train_layer3(
    obs_df: pd.DataFrame,
    price_data: pd.DataFrame,
    train_dates: List[str],
    all_tickers: List[str],
    seeds: List[int] = [1,2,3,4,5],
    num_days: int = 10,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    n_epochs: int = 2000,
    save_dir: str = "./models/layer3"
) -> Dict[str, Any]:
    """
    Train layer 3 RL agents with attention mechanism.
    """
    os.makedirs(save_dir, exist_ok=True)
    trained_models = {}
    
    for seed in seeds:
        print(f"\n==== Training Layer 3 Seed {seed} ====")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        env = OptionsTradingEnv(
            all_tickers=all_tickers,
            obs_df=obs_df,
            price_data=price_data,
            train_dates=train_dates,
            num_days=num_days
        )
        
        buffer = ReplayBuffer(max_size=50000)
        
        # Create network
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = Layer3ActorCritic(obs_dim, action_dim)
        optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
        
        # Training loop
        episode_rewards = []
        for epoch in tqdm(range(n_epochs), desc=f"Layer 3 Training"):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    action, _ = agent.get_action(torch.FloatTensor(obs).unsqueeze(0))
                    action = action.squeeze(0).numpy()
                
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                buffer.add(obs, action, reward, next_obs, done)
                obs = next_obs
                
                if len(buffer) >= batch_size:
                    experiences = buffer.sample(batch_size)
                    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = experiences
                    
                    # Get policy outputs
                    mean, std, value = agent(obs_batch)
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                    
                    # Compute target value
                    with torch.no_grad():
                        _, _, next_value = agent(next_obs_batch)
                        target_value = reward_batch + (1 - done_batch) * 0.99 * next_value
                    
                    # Value loss
                    value_loss = nn.MSELoss()(value, target_value)
                    
                    # Policy loss
                    advantage = (target_value - value).detach()
                    policy_loss = -(log_prob * advantage).mean()
                    
                    # Total loss
                    loss = value_loss + policy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            episode_rewards.append(episode_reward)
            
            if (epoch + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Epoch {epoch+1}, Average Reward: {avg_reward:.2f}")
        
        # Save model
        model_dir = os.path.join(save_dir, f"seed_{seed}")
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save(agent.state_dict(), os.path.join(model_dir, "agent.pt"))
        
        trained_models[seed] = agent
        
        # Cleanup
        del agent, optimizer, env
        gc.collect()
        torch.cuda.empty_cache()
    
    return trained_models