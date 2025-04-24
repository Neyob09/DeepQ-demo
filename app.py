
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

st.set_page_config(page_title="2D Bin Packing Demo", layout="wide")

# --- Bin Packing Environment ---
class BinPacking2DEnv:
    def __init__(self, bin_width=1.0, bin_height=1.0, num_bins=5, items=None):
        self.bin_width = bin_width
        self.bin_height = bin_height
        self.num_bins = num_bins
        self.items = items
        self.bins = [[] for _ in range(self.num_bins)]
        self.current_index = 0
        self.current_item = self.items[self.current_index]
    
    def reset(self):
        self.bins = [[] for _ in range(self.num_bins)]
        self.current_index = 0
        self.current_item = self.items[self.current_index]
        return self.get_state()

    def step(self, action):
        reward = -1
        item = self.current_item
        if self._can_place(self.bins[action], item):
            self.bins[action].append(item)
            reward = 1
        self.current_index += 1
        done = self.current_index >= len(self.items)
        if not done:
            self.current_item = self.items[self.current_index]
        return self.get_state(), reward, done

    def _can_place(self, bin_items, item):
        y_offset = sum(h for _, h in bin_items)
        return y_offset + item[1] <= self.bin_height and item[0] <= self.bin_width

    def get_state(self):
        flat = [d for bin in self.bins for r in bin for d in r][:30]
        padded = flat + [0.0] * (30 - len(flat))
        return np.array(list(self.current_item) + padded)

    def get_bins_used(self):
        return sum(1 for b in self.bins if b)

# --- Vanilla DQN ---
class VanillaDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.model.predict(state, verbose=0)
        return np.argmax(q[0])

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for s, a, r, s_, done in minibatch:
            target = self.model.predict(s, verbose=0)
            if done:
                target[0][a] = r
            else:
                target[0][a] = r + self.gamma * np.amax(self.model.predict(s_, verbose=0)[0])
            self.model.fit(s, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Heuristics ---
def heuristic_2d(items, strategy="first_fit"):
    bins = [[]]
    for item in items:
        placed = False
        if strategy == "best_fit":
            best_index = -1
            best_height = 1.1
            for i, b in enumerate(bins):
                height = sum(h for _, h in b)
                if height + item[1] <= 1.0 and height < best_height:
                    best_index = i
                    best_height = height
            if best_index != -1:
                bins[best_index].append(item)
                placed = True
        else:
            for b in bins:
                height = sum(h for _, h in b)
                if height + item[1] <= 1.0:
                    b.append(item)
                    placed = True
                    break
        if not placed:
            bins.append([item])
    return bins

# --- Streamlit UI ---
st.title("🏭 Warehouse Bin Packing Assistant")

mode = st.radio("Choose input mode:", ["Generate Random Items", "Enter Items Manually"])

if mode == "Generate Random Items":
    num_items = st.slider("Number of items", 5, 30, 10)
    min_size = st.slider("Min item size", 0.05, 0.3, 0.1)
    max_size = st.slider("Max item size", 0.2, 0.5, 0.4)
    items = [tuple(np.round(np.random.uniform(min_size, max_size, size=2), 2)) for _ in range(num_items)]
else:
    num_items = st.slider("Number of manual items", 5, 20, 10)
    items = []
    st.write("Enter Width and Height for each item:")
    for i in range(num_items):
        cols = st.columns(2)
        w = cols[0].number_input(f"Item {i+1} Width", min_value=0.05, max_value=1.0, value=0.1, key=f"w_{i}")
        h = cols[1].number_input(f"Item {i+1} Height", min_value=0.05, max_value=1.0, value=0.1, key=f"h_{i}")
        items.append((round(w, 2), round(h, 2)))

st.write("🧾 Item list:", items)

# Evaluate heuristics
ff_bins = heuristic_2d(items, strategy="first_fit")
bf_bins = heuristic_2d(items, strategy="best_fit")

# Vanilla DQN
env = BinPacking2DEnv(items=items)
state_size = len(env.get_state())
vanilla = VanillaDQN(state_size, env.num_bins)
state = env.reset().reshape(1, state_size)
done = False
while not done:
    action = vanilla.act(state)
    next_state, reward, done = env.step(action)
    vanilla.remember(state, action, reward, next_state.reshape(1, state_size), done)
    state = next_state.reshape(1, state_size)
if len(vanilla.memory) >= 32:
    vanilla.replay(32)
vanilla_bins = env.get_bins_used()

# Double DQN
try:
    model = load_model("double_dqn_episode25.keras", compile=False)
    env = BinPacking2DEnv(items=items)
    state = env.reset().reshape(1, state_size)
    done = False
    while not done:
        action = np.argmax(model.predict(state, verbose=0)[0])
        state, _, done = env.step(action)
        state = state.reshape(1, state_size)
    ddqn_bins = env.get_bins_used()
except Exception as e:
    ddqn_bins = "❌ Model not found or invalid"

st.subheader("📊 Bins Used")
st.write(f"🧠 Double DQN: {ddqn_bins}")
st.write(f"🧠 Vanilla DQN: {vanilla_bins}")
st.write(f"📦 First Fit: {len(ff_bins)}")
st.write(f"📦 Best Fit: {len(bf_bins)}")
