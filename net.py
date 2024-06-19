import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Parámetros del entorno
IMAGE_SHAPE = (480, 640, 3)
ACTION_SPACE = ['up', 'down', 'left', 'right']  # Ejemplo de acciones
STATE_SIZE = (480, 640, 3)  # Tamaño del estado
ACTION_SIZE = len(ACTION_SPACE)  # Número de acciones posibles
MEMORY_SIZE = 10000  # Tamaño del buffer de memoria
BATCH_SIZE = 64  # Tamaño del lote de entrenamiento
GAMMA = 0.99  # Factor de descuento
ALPHA = 0.001  # Tasa de aprendizaje

# Buffer de memoria para almacenar experiencias
memory = deque(maxlen=MEMORY_SIZE)

# Inicialización del puntaje de recompensa
reward = 0

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_size[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 22 * 31, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Reordenar dimensiones para convolución
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Crear la red neuronal
model = DQN(STATE_SIZE, ACTION_SIZE)
target_model = DQN(STATE_SIZE, ACTION_SIZE)
target_model.load_state_dict(model.state_dict())
target_model.eval()

# Definir el optimizador
optimizer = optim.Adam(model.parameters(), lr=ALPHA)

# Definir la función de pérdida
criterion = nn.MSELoss()


def select_action(state, epsilon):
    if random.random() <= epsilon:
        return random.choice(ACTION_SPACE)
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state)
        return ACTION_SPACE[torch.argmax(q_values).item()]

def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def train_model():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(states)
    actions = torch.LongTensor([ACTION_SPACE.index(a) for a in actions]).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)
    
    current_q_values = model(states).gather(1, actions)
    next_q_values = target_model(next_states).max(1)[0].unsqueeze(1)
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
    
    loss = criterion(current_q_values, expected_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


