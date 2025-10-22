import sys
import os

# Add the directory containing the .pyd file to the Python path
# This is necessary so Python can find our Rust module.
# The path is relative to this script's location.
module_path = os.path.join(os.path.dirname(__file__), "nexus", "target", "release")
sys.path.append(module_path)

try:
    import nexus
    print("Successfully imported 'nexus' module from Rust.")
except ImportError as e:
    print(f"Failed to import 'nexus' module: {e}")
    print(f"Please ensure '{os.path.join(module_path, 'nexus.pyd')}' exists.")
    sys.exit(1)


class MyPythonAgent:
    """
    A simple agent defined in Python that implements the required `id` and `on_tick` methods.
    """
    def __init__(self, agent_id):
        self.id = agent_id
        self.tick_count = 0

    def on_tick(self, market_state):
        """
        This method is called from the Rust simulation on every tick.
        """
        print(f"[Python Agent {self.id}] Tick {self.tick_count}. Market sequence: {market_state.sequence}")
        print(f"[Python Agent {self.id}] Bids: {market_state.bids}, Asks: {market_state.asks}")
        
        self.tick_count += 1
        
        # On the first tick, submit a buy order.
        if self.tick_count == 1:
            print(f"[Python Agent {self.id}] Submitting a buy order.")
            # nexus.Order(agent_id, price, size, is_bid)
            return [nexus.Order(self.id, 100.0, 5.0, True)]
            
        # On the third tick, submit a sell order that should match the first order.
        if self.tick_count == 3:
            print(f"[Python Agent {self.id}] Submitting a sell order.")
            return [nexus.Order(self.id, 100.0, 2.0, False)]

        # On all other ticks, do nothing.
        return []


def main():
    print("\n--- Setting up simulation ---")
    # Create a simulation instance from our Rust code
    sim = nexus.Simulation()

    # Create two Python agents
    agent1 = MyPythonAgent(agent_id=1)
    agent2 = MyPythonAgent(agent_id=2)

    # Add the Python agents to the Rust simulation
    sim.add_agent(agent1)
    sim.add_agent(agent2)
    print("Added 2 Python agents to the Rust simulation.")

    print("\n--- Running simulation for 5 ticks ---")
    for i in range(5):
        print(f"\n--- Tick {i+1} ---")
        trades = sim.tick()
        if trades:
            print(f"Trades occurred in tick {i+1}:")
            for trade in trades:
                print(f"  - Price: {trade.price}, Size: {trade.size}, Aggressor: {trade.aggressor_agent_id}, Resting: {trade.resting_agent_id}")
        else:
            print("No trades occurred.")
            
    print("\n--- Simulation finished ---")
    final_market_state = sim.get_market_state()
    print(f"Final Bids: {final_market_state.bids}")
    print(f"Final Asks: {final_market_state.asks}")


if __name__ == "__main__":
    main()
