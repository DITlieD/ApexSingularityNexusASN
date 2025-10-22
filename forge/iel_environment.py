import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

logger = logging.getLogger("IEL_Env")

# Configuration (Must match Rust implementation)
# State Space: Imbalance, Norm Spread, Pressure, Signal Strength/Side
STATE_DIM = 4
# Action Space: 0=Aggressive (Taker), 1=Passive (Maker at Touch), 2=Deep Passive
ACTION_DIM = 3

class LOBExecutionEnv(gym.Env):
    """
    A simplified environment simulating LOB dynamics for execution optimization.
    The goal is to maximize reward (minimize execution cost).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(LOBExecutionEnv, self).__init__()
        self.action_space = spaces.Discrete(ACTION_DIM)
        # State space normalized between -1.0 and 1.0
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
        self.current_state = None
        self.signal_side = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Simulate a random market state
        self.current_state = self.observation_space.sample()
        # The last element (index 3) is the signal strength/side
        self.signal_side = 1 if self.current_state[3] > 0 else -1
        return self.current_state, {}

    def step(self, action):
        imbalance = self.current_state[0]
        # Spread (index 1) is normalized (0 to 1). Ensure it's positive.
        spread = max(0, self.current_state[1]) 
        
        cost = 0.0
        # Constants based on Bybit fees (used in previous steps)
        MAKER_FEE = 0.0002
        TAKER_FEE = 0.00055

        # Define Costs and Fill Probabilities based on Tactic
        if action == 0: # Aggressive (Taker)
            # Slippage increases with spread and if imbalance opposes the signal
            adverse_imbalance = (0.0002 if imbalance * self.signal_side < 0 else 0)
            slippage = spread * 0.001 + adverse_imbalance
            cost = TAKER_FEE + slippage
            fill_prob = 0.99

        elif action == 1: # Passive (Maker at Touch)
            cost = MAKER_FEE
            # Fill probability depends on imbalance favoring the signal
            fill_prob = 0.5 + imbalance * self.signal_side * 0.4

        elif action == 2: # Deep Passive
            # Price improvement (negative slippage)
            slippage = -0.0001 
            cost = MAKER_FEE + slippage
            fill_prob = 0.2 + imbalance * self.signal_side * 0.1

        # Simulate fill
        filled = np.random.rand() < fill_prob
        
        # Reward function: Minimize expected cost
        # Opportunity Cost: Penalty for missing the trade (scaled by signal strength).
        opportunity_cost = abs(self.current_state[3]) * 0.0015

        if filled:
            reward = -cost
        else:
            reward = -opportunity_cost
            
        # The episode ends after one execution attempt
        terminated = True
        return self.current_state, reward, terminated, False, {}