import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# =========================
# Hyperparameters
# =========================
N_AGENTS = 5
STATE_DIM = 5
ACTION_DIM = 1

GAMMA = 0.95
TAU = 0.01
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3

BUFFER_SIZE = 50000
BATCH_SIZE = 64

EPISODES = 200
STEPS = 20

DEVICE = torch.device("cpu")


# =========================
# Environment
# =========================
class EdgeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.d = np.random.uniform(0.5, 2, N_AGENTS)
        self.C = np.random.uniform(1e8, 5e8, N_AGENTS)
        self.f = np.random.uniform(1e9, 2e9, N_AGENTS)
        self.queue = np.zeros(N_AGENTS)
        return self._state()

    def _state(self):
        states = []
        for i in range(N_AGENTS):
            sinr = np.random.rand()
            states.append(np.array([sinr, self.queue[i], self.f[i], self.d[i], self.C[i]]))
        return np.array(states, dtype=np.float32)

    def step(self, actions):
        rewards = []
        delays = []
        energies = []

        for i in range(N_AGENTS):
            a = actions[i]

            tx_delay = self.d[i] / (1 + np.random.rand())
            local_delay = self.C[i] / self.f[i]
            queue_delay = self.queue[i] * 0.1

            edge_delay = tx_delay + queue_delay + self.C[i] / 1e9
            delay = a * edge_delay + (1 - a) * local_delay

            energy = a * 2 + (1 - a) * 1

            reward = -(0.6 * delay + 0.4 * energy)

            self.queue[i] = max(0, self.queue[i] + np.random.randn() * 0.1)

            rewards.append(reward)
            delays.append(delay)
            energies.append(energy)

        return self._state(), rewards, np.mean(delays), np.mean(energies)


# =========================
# Networks
# =========================
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM * N_AGENTS + N_AGENTS, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.net(x)


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def push(self, s, a, r, s2):
        self.buffer.append((s, a, r, s2))

    def sample(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        s, a, r, s2 = map(np.array, zip(*batch))
        return s, a, r, s2

    def size(self):
        return len(self.buffer)


# =========================
# MADDPG Agent
# =========================
class MADDPG:
    def __init__(self):
        self.actors = [Actor().to(DEVICE) for _ in range(N_AGENTS)]
        self.critics = [Critic().to(DEVICE) for _ in range(N_AGENTS)]

        self.target_actors = [Actor().to(DEVICE) for _ in range(N_AGENTS)]
        self.target_critics = [Critic().to(DEVICE) for _ in range(N_AGENTS)]

        for i in range(N_AGENTS):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        self.actor_opt = [optim.Adam(a.parameters(), lr=LR_ACTOR) for a in self.actors]
        self.critic_opt = [optim.Adam(c.parameters(), lr=LR_CRITIC) for c in self.critics]

        self.buffer = ReplayBuffer()

    def select_action(self, s):
        actions = []
        for i in range(N_AGENTS):
            state = torch.FloatTensor(s[i]).unsqueeze(0)
            a = self.actors[i](state).detach().numpy()[0][0]
            actions.append(a)
        return actions

    def update(self):
        if self.buffer.size() < BATCH_SIZE:
            return

        s, a, r, s2 = self.buffer.sample()

        s = torch.FloatTensor(s.reshape(BATCH_SIZE, -1))
        s2 = torch.FloatTensor(s2.reshape(BATCH_SIZE, -1))
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r)

        for i in range(N_AGENTS):
            # critic update
            with torch.no_grad():
                a2 = torch.cat([
                    self.target_actors[j](torch.FloatTensor(s2[:, j*STATE_DIM:(j+1)*STATE_DIM]))
                    for j in range(N_AGENTS)
                ], dim=1)

                q_target = r[:, i].unsqueeze(1) + GAMMA * self.target_critics[i](s2, a2)

            q_val = self.critics[i](s, a)
            critic_loss = ((q_val - q_target) ** 2).mean()

            self.critic_opt[i].zero_grad()
            critic_loss.backward()
            self.critic_opt[i].step()

            # actor update
            a_pred = torch.cat([
                self.actors[j](s[:, j*STATE_DIM:(j+1)*STATE_DIM])
                for j in range(N_AGENTS)
            ], dim=1)

            actor_loss = -self.critics[i](s, a_pred).mean()

            self.actor_opt[i].zero_grad()
            actor_loss.backward()
            self.actor_opt[i].step()

            # soft update
            for p, tp in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

            for p, tp in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)


# =========================
# Training
# =========================
env = EdgeEnv()
agent = MADDPG()

reward_list = []
delay_list = []
energy_list = []

for ep in range(EPISODES):
    s = env.reset()

    ep_reward = 0
    ep_delay = 0
    ep_energy = 0

    for t in range(STEPS):
        a = agent.select_action(s)

        s2, r, d, e = env.step(a)

        agent.buffer.push(s, a, r, s2)
        agent.update()

        s = s2

        ep_reward += np.mean(r)
        ep_delay += d
        ep_energy += e

    reward_list.append(ep_reward)
    delay_list.append(ep_delay / STEPS)
    energy_list.append(ep_energy / STEPS)

    print(f"Episode {ep} | Reward {ep_reward:.3f} | Delay {ep_delay/STEPS:.3f} | Energy {ep_energy/STEPS:.3f}")


# =========================
# Plot results
# =========================
# =========================
# Plot results
# =========================

plt.figure()
plt.plot(reward_list)
plt.title("Training Reward Convergence")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.savefig("reward_convergence.png", dpi=300, bbox_inches="tight")


plt.figure()
plt.plot(delay_list)
plt.title("Average Delay")
plt.xlabel("Episode")
plt.ylabel("Delay")

plt.savefig("delay_curve.png", dpi=300, bbox_inches="tight")


plt.figure()
plt.plot(energy_list)
plt.title("Average Energy")
plt.xlabel("Episode")
plt.ylabel("Energy")

plt.savefig("energy_curve.png", dpi=300, bbox_inches="tight")

plt.show()