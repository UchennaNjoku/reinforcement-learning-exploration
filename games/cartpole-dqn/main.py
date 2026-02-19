import random
import argparse
from collections import deque, namedtuple

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

# === NEURAL NET ==========================================================================
# our input of four observations gets passed to 128 neurons then an activation function
# the 128 gets passed to another hidden layer of 128 then finally forward to our 
# two actions   # NB. calling this (passing in a state tensor) returns q values
class QNetwork(nn.Module):
    def __init__(self, observation_size: int, num_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,num_actions)
        )
    
    def forward(self, x):
        return self.network(x)
    
# === REPLAY BUFFER ==========================================================================

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"]) # for readability

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        t = Transition(state, action, reward, next_state, done)
        self.buffer.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch)) # converts list of Transitions into a single transition of lists (grouping the corresponding elements)
    
    def __len__(self):
        return len(self.buffer)
    
# === ACTION SELECTION (USING EPSILON-GREEDY) ===============================================
def select_action(neural_net, state, epsilon, num_actions, device):
    # Explore
    if (random.random() < epsilon):
        return random.randrange(num_actions)
    # Exploit
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = neural_net(s)
        return int(torch.argmax(q_values, dim=1).item())
    
# === TRAINING STEP ==========================================================================
def update_network(q_network, target_network, optimizer, batch, gamma, device):
    # Convert batch -> tensors
    s  = torch.tensor(np.array(batch.state),  dtype=torch.float32, device=device)  # (B,4)
    a  = torch.tensor(np.array(batch.action),            dtype=torch.int64,   device=device)  # (B,)
    r  = torch.tensor(np.array(batch.reward),            dtype=torch.float32, device=device)  # (B,)
    s2 = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)  # (B,4)
    d  = torch.tensor(np.array(batch.done),         dtype=torch.float32, device=device)  # (B,) 1 if done else 0

    # === PREDICTION (from online network)
    # we first find all q values for each state in our batch
    # then we get the q val for the chosen action
    q_all = q_network(s)
    q_val_chosen_action = q_all.gather(1, a.unsqueeze(1))
    q_val_chosen_action = q_val_chosen_action.squeeze(1) # remove extra dimension

    # === TARGET (from target network)
    # target formula -> target [y] is the reward you got + discounted best future q (iff episode not done)
    with torch.no_grad():
        q_target_all = target_network(s2)
        max_q_target = torch.max(q_target_all, dim=1).values
        
        y = r + gamma * (1.0 - d) * max_q_target # NB. if done then the target is just the reward

    # === LOSS 
    loss = torch.mean((y - q_val_chosen_action) ** 2)

    optimizer.zero_grad() # clear old gradients
    loss.backward() # backprop (for each weight how should it change to reduce loss)
    optimizer.step() # update weights 

    return float(loss.item()) # for logging float if needed

def train_cartpole_dqn(
        env_name="CartPole-v1",
        seed=0,
        episodes=600,
        gamma=0.99,
        lr=1e-3,
        buffer_size=50_000,
        batch_size=64,
        warmup_steps=1_000,
        target_update_every=500,   # steps
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=30_000,
        max_steps_per_episode=500, 
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = gym.make(env_name)
    observation_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    q_net = QNetwork(observation_size, num_actions).to(device)
    target_net = QNetwork(observation_size, num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict()) # copy weights so they are identical
    target_net.eval() # set to evaluation mode (removing features like dropout, not needed but standard practice)

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)

    num_global_steps = 0 # total environment steps taken across all episode
    total_reward_per_episode = [] 

    def epsilon_by_step(t):
        frac = min(1.0, t / epsilon_decay_steps)
        return epsilon_start + frac * (epsilon_end - epsilon_start)
    
    for ep in range(episodes):
        state, info = env.reset(seed=seed + ep)
        ep_return = 0.0

        for _ in range(max_steps_per_episode):
            eps = epsilon_by_step(num_global_steps)
            action = select_action(q_net, state, eps, num_actions, device)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            ep_return += reward
            num_global_steps += 1

            # TRAIN ONLY AFTER WARMUP STEPS
            if num_global_steps >= warmup_steps and len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                _loss = update_network(q_net, target_net, optimizer, batch, gamma, device)

                if num_global_steps % target_update_every == 0:
                    target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        total_reward_per_episode.append(ep_return)
        recent = total_reward_per_episode[-50:]
        avg50 = np.mean(recent)
        print(f"Ep {ep:4d} | Return {ep_return:6.1f} | Avg50 {avg50:6.1f} | eps {epsilon_by_step(num_global_steps):.3f}")

        if len(total_reward_per_episode) >= 100 and np.mean(total_reward_per_episode[-100:]) >= 475:
            print("Solved (avg >= 475 over last 100).")
            break

    env.close()
    return q_net

# WATCH TRAINED AGENT
def watch_agent(q_net, episodes=5, env_name="CartPole-v1"):
    env = gym.make(env_name, render_mode="human")
    device = next(q_net.parameters()).device
    for ep in range(episodes):
        state, info = env.reset()
        total = 0
        done = False

        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q = q_net(s)  # (1,2)
                action = int(torch.argmax(q, dim=1).item())

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += reward

        print(f"[WATCH] Episode {ep} return = {total}")

    env.close()


# === PLAY MANUALLY (console input) =========================
def play_manual_console(env_name="CartPole-v1", episodes=3):
    """
    Manual control via console:
    - type 0 for left
    - type 1 for right
    """
    env = gym.make(env_name, render_mode="human")

    for ep in range(episodes):
        state, info = env.reset()
        total = 0
        done = False
        print("\nManual play: type 0 (left) or 1 (right). Enter to submit.")

        while not done:
            # Get user action
            a_str = input("Action [0/1]: ").strip()
            if a_str not in ("0", "1"):
                print("Please type 0 or 1.")
                continue
            action = int(a_str)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += reward

        print(f"[MANUAL] Episode {ep} return = {total}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN CartPole — train, watch, or play manually")
    parser.add_argument("mode", 
                        choices=["train", "watch", "manual"],
                        help="train: train the DQN agent | watch: load & watch a trained agent | manual: play yourself")
    parser.add_argument("--episodes", type=int, default=600, help="number of episodes (default: 600)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--model-path", type=str, default="cartpole_dqn.pt",
                        help="path to save/load model weights (default: cartpole_dqn.pt)")
    args = parser.parse_args()

    if args.mode == "train":
        trained_net = train_cartpole_dqn(episodes=args.episodes, lr=args.lr, seed=args.seed)
        torch.save(trained_net.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

    elif args.mode == "watch":
        # CartPole-v1: 4 observations, 2 actions
        q_net = QNetwork(4, 2)
        q_net.load_state_dict(torch.load(args.model_path, weights_only=True))
        q_net.eval()
        print(f"Loaded model from {args.model_path}")
        watch_agent(q_net, episodes=args.episodes)

    elif args.mode == "manual":
        play_manual_console(episodes=args.episodes)

