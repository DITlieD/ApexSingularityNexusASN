import operator
import random
import numpy
import pandas as pd
import polars as pl # Import Polars
import time
import json
import os

from deap import algorithms, base, creator, tools, gp

# Import the engines and managers
from causal_engine import perform_causal_discovery
from chimera_engine import find_dsg
from fitness_ttt import evaluate_individual
from serialize_strategy import deap_to_json
from validation_gauntlet import ValidationGauntlet # NEW IMPORT
# (Add necessary imports at the top)
from deap import gp
import polars as pl
import numpy as np
import time
import os
# Import MAML components
from maml_adapter import MAMLManager, TaskGenerator, CONTEXT_SIZE, FEATURE_SIZE

# (Ensure POTENTIAL_FEATURES matches FEATURE_SIZE=5 and TARGET_VARIABLE is defined)


# NEW HELPER: Function to execute GP on historical data
def execute_gp_strategy(individual, data_pl: pl.DataFrame, feature_names: list[str], pset: gp.PrimitiveSet):
    """Executes the GP strategy (Signal and Size) on the historical data."""
    
    # Prepare data (We convert Polars Decimal data to Pandas/Numpy float32 for execution)
    data_pd = data_pl.select(feature_names).to_pandas().astype(np.float32)
    if data_pd.empty:
        return pl.DataFrame({'gp_signal': [], 'gp_size': []})

    input_args = [data_pd[name].values for name in feature_names]

    def execute_tree(tree):
        try:
            func = gp.compile(expr=tree, pset=pset)
            result = func(*input_args)
            # Handle scalar outputs
            if np.isscalar(result):
                return np.full(len(data_pd), result)
            return np.array(result).flatten()
        except Exception as e:
            return np.zeros(len(data_pd))

    # GP individual structure: [SignalTree, SizeTree]
    signal = execute_tree(individual[0])
    size = execute_tree(individual[1])
    
    # Apply clamping (as done in Rust interpreter)
    size = np.clip(np.abs(size), 0.001, 0.5)

    # Return as Polars DataFrame, ensuring Decimal type for consistency with the rest of the Forge
    return pl.DataFrame({'gp_signal': signal, 'gp_size': size}).with_columns(
        pl.all().cast(pl.Decimal(scale=8, precision=None))
    )

# --- Helper Functions (Primitives) ---
def protectedDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1

def if_then_else(input, output1, output2):
    return output1 if input > 0 else output2

# --- 2. Create the Types (Ensure creation only once) ---
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax) # Dual-Tree


# --- 3. Dynamic Toolbox Setup (The Core ACN Integration) ---

# (Helper functions for dual tree initialization and mutation from Part 1)
def initDualTree(container, func):
    return container([gp.PrimitiveTree(func()) for _ in range(2)])

def cxOnePointDual(ind1, ind2):
    if random.random() < 0.7: gp.cxOnePoint(ind1[0], ind2[0])
    if random.random() < 0.7: gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

def mutUniformDual(individual, expr, pset):
    if random.random() < 0.5: gp.mutUniform(individual[0], expr=expr, pset=pset)
    if random.random() < 0.5: gp.mutUniform(individual[1], expr=expr, pset=pset)
    return individual,

def combined_height(ind):
    return max(tree.height for tree in ind)


def setup_dynamic_toolbox(causal_features, dsg, historical_data_pl: pl.DataFrame, current_onnx_path: str=None):
    """
    Dynamically configures the GP toolbox based on causal analysis and the DSG.
    """
    toolbox = base.Toolbox()
    
    # 1. Create the filtered Primitive Set (Causality)
    pset = gp.PrimitiveSet("MAIN", len(causal_features))
    
    # Add Primitives (Operators)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(protectedDiv, 2, name="protectedDiv")
    pset.addPrimitive(if_then_else, 3, name="ifThenElse")
    pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

    # Rename the ARGs (ARG0, ARG1...) to the names of the causal features
    rename_map = {f"ARG{i}": name for i, name in enumerate(causal_features)}
    pset.renameArguments(**rename_map)

    # 2. Register GP operators
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", initDualTree, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 3. Register Evaluation (Dual-Fitness)
    # We pass the historical data and feature names required by the Causal Fitness backtester
    toolbox.register("evaluate", evaluate_individual, 
                     dsg=dsg, 
                     historical_data_pl=historical_data_pl, 
                     feature_names=causal_features,
                     onnx_path=current_onnx_path) # NEW

    # 4. Register Selection and Mutation
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", cxOnePointDual)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", mutUniformDual, expr=toolbox.expr_mut, pset=pset)

    # Decorators
    toolbox.decorate("mate", gp.staticLimit(key=combined_height, max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=combined_height, max_value=8))

    return toolbox, pset

# --- Main Execution (The Forge Cycle) ---
def main(historical_data_path=None, run_chimera=True, use_sae=True):
    print("--- ASN Forge Cycle Start (ACN Synthesis) ---")

    # 1. Data Preparation (Using Polars for Rust compatibility)
    print("1. Preparing Data...")
    if historical_data_path is None:
        print("1. Preparing Synthetic Data (Placeholder)...")
        T = 5000
        df = pd.DataFrame(numpy.random.randn(T, 3), columns=POTENTIAL_FEATURES)
        # Create a target variable influenced by 'imbalance'
        df['target_return'] = df['imbalance'].shift(-1) + numpy.random.randn(T) * 0.1
        df['close'] = 100 + df['target_return'].cumsum()
        df = df.dropna()
        data = pl.from_pandas(df)
    else:
        # This is a placeholder for loading real data
        # For example: data = pl.read_csv(historical_data_path)
        # Using synthetic data for now.
        print("1. Preparing Synthetic Data (Placeholder)...")
        T = 5000
        df = pd.DataFrame(numpy.random.randn(T, 3), columns=POTENTIAL_FEATURES)
        df['target_return'] = df['imbalance'].shift(-1) + numpy.random.randn(T) * 0.1
        df['close'] = 100 + df['target_return'].cumsum()
        df = df.dropna()
        data = pl.from_pandas(df)


    # CRITICAL: Ensure data types are Decimal for Rust Backtester compatibility
    decimal_cols = POTENTIAL_FEATURES + [TARGET_VARIABLE, 'close']
    data = data.with_columns([
        pl.col(col).cast(pl.Decimal(scale=8, precision=None)) for col in decimal_cols if col in data.columns
    ])


    # 2. Causal Discovery (Tigramite requires Pandas/Numpy and Floats)
    data_pd = data.to_pandas()
    # Convert Decimal columns to float for Tigramite compatibility
    for col in data_pd.columns:
        if isinstance(data_pd[col].iloc[0], object): # Simple check for Decimal objects
            data_pd[col] = data_pd[col].astype(float)

    print("\n2. Running Causal Discovery...")
    causal_features = perform_causal_discovery(
        data_pd[POTENTIAL_FEATURES + ['target_return']], 
        target_variable='target_return',
        tau_max=5,
        pc_alpha=0.01
    )
    
    # Ensure we only use features available in the Rust simulator
    available_causal_features = [f for f in causal_features if f in POTENTIAL_FEATURES]
    if not available_causal_features:
        print("WARNING: No causal features found among available Rust features. Falling back to defaults.")
        available_causal_features = POTENTIAL_FEATURES


    # 3. Chimera Engine (Find DSG)
    if run_chimera:
        print("\n3. Running Chimera Engine (Inference)...")
        dsg = find_dsg(sigma0=0.2, maxiter=30) # Reduced iterations for speed
    else:
        # Load DSG from file if available
        if os.path.exists("current_dsg.json"):
             with open("current_dsg.json", 'r') as f:
                dsg = json.load(f).get("dsg")
             print("\n3. Loaded DSG from file.")
        else:
            dsg = None

    if dsg is None:
        print("WARNING: Chimera Engine failed or skipped. Using fallback DSG.")
        # MODIFIED: Default market making parameters (4 parameters)
        dsg = [0.5, 0.01, 0.0001, 1.0] 

    # Initialize current_onnx_path (used if evolving an existing hybrid strategy)
    current_onnx_path = None 

    # 4. Setup Dynamic Toolbox
    # ... (Data preparation for Rust)
    
    # Pass the current ONNX path to the toolbox setup
    toolbox, pset = setup_dynamic_toolbox(available_causal_features, dsg, historical_data_pl, current_onnx_path)

    # 5. Evolution (Hybrid Crucible with SAE)
    
    # SAE Configuration
    POP_SIZE = 500 if use_sae else 100
    NGEN = 25
    TOP_K_PERCENT = 0.05 # Evaluate top 5%

    print(f"\n--- Starting GP Evolution (Synthesis) ---")
    print(f"SAE Enabled: {use_sae}. Pop Size: {POP_SIZE}. Generations: {NGEN}.")

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", numpy.mean)
    mstats.register("max", numpy.max)


    try:
        if use_sae:
             # Initialize SAE components
            vectorizer = DNAVectorizer(pset)
            oracle = FitnessOracle(vectorizer)
            sae_manager = SurrogateAssistedEvolution(oracle, top_k_percent=TOP_K_PERCENT)
            
            # Run the custom SAE evolutionary loop
            run_evolutionary_loop(pop, toolbox, NGEN, mstats, hof, sae_manager)
        else:
            # Run standard evolution
            algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.3, ngen=NGEN, stats=mstats, halloffame=hof, verbose=True)
    except Exception as e:
         print(f"\nERROR during evolution. Ensure Rust 'nexus' module is compiled and accessible. Details: {e}")
         return


    # 6. MAML Training and Validation (Updated Integration)
    if hof and len(hof) > 0:
        best_ind = hof[0]
        print(f"\nEvolution Complete. Best Fitness (Combined TTT Score): {best_ind.fitness.values[0]:.2f}")
        
        timestamp = int(time.time())
        onnx_filename = f"maml_modulator_{timestamp}.onnx"
        maml_path = None # Initialize maml_path

        # A. Generate GP Outputs (Needed for MAML training AND Validation)
        print("\n--- Generating GP Outputs (Historical) ---")
        # We use maml_data_pl if available (ensures feature alignment), otherwise fallback to historical_data_pl
        data_source = maml_data_pl if 'maml_data_pl' in locals() and maml_data_pl is not None else historical_data_pl

        if data_source is not None:
            try:
                # Execute the best GP strategy on the historical data
                gp_outputs = execute_gp_strategy(best_ind, data_source, available_causal_features, pset)
            except Exception as e:
                print(f"ERROR generating GP outputs: {e}")
                gp_outputs = None
        else:
            gp_outputs = None

        # B. MAML Training (Only if data and GP outputs are available)
        if 'maml_data_pl' in locals() and maml_data_pl is not None and gp_outputs is not None:
            print("\n--- Starting MAML Modulation Training ---")
            try:
                task_generator = TaskGenerator(maml_data_pl, gp_outputs, available_causal_features, TARGET_VARIABLE)
                maml_manager = MAMLManager()
                maml_manager.meta_train(task_generator, iterations=500) 
                maml_manager.export_onnx(onnx_filename)
                maml_path = onnx_filename
            except Exception as e:
                print(f"ERROR during MAML training phase: {e}")
                if os.path.exists(onnx_filename):
                    os.remove(onnx_filename)
        
        # 7. Validation Gauntlet (Updated Integration)
        print("\n--- Starting Validation Gauntlet ---")
        
        # We must have the historical data (with 'close') and the GP outputs to run validation.
        # historical_data_pl is used for features/close prices.
        if historical_data_pl is not None and gp_outputs is not None:
             
            # Initialize the Gauntlet with all required components
            gauntlet = ValidationGauntlet(
                data, # Pass the full dataset with target_return
                available_causal_features, 
                pset, 
                gp_outputs, # Pass the pre-computed GP outputs
                maml_path     # Pass the path to the ONNX file (can be None)
            )
            # The Gauntlet handles the hybrid execution internally.
            validation_passed = gauntlet.run(best_ind)
        else:
            print("ERROR: Missing historical data or GP outputs for validation. Forcing FAIL.")
            validation_passed = False


        # (Deployment logic remains the same)
        if validation_passed:
            print("\n--- GAUNTLET PASSED: Deploying Hybrid Strategy ---")
            # Save the GP strategy
            json_output = deap_to_json(best_ind)
            strategy_filename = f"strategy_challenger_{timestamp}.json"
            with open(strategy_filename, "w") as f:
                f.write(json_output)
            
            # The ONNX file is already saved. The Pit Crew must detect BOTH files.
            print(f"Deployed GP: '{strategy_filename}'. MAML (if generated): '{maml_path}'")
        else:
            print("\n--- GAUNTLET FAILED: Strategy Discarded ---")
            # If validation fails, remove the associated ONNX file if it exists
            if maml_path and os.path.exists(maml_path):
                print(f"Removing associated MAML file: {maml_path}")
                os.remove(maml_path)

    print("\n--- ASN Forge Cycle Complete ---")
    return pop, mstats, hof

# NEW: Unified Evolutionary Loop (Handles both Standard and SAE)
def run_evolutionary_loop(population, toolbox, ngen, stats, halloffame, sae_manager=None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the initial population entirely (Required for both standard and seeding SAE Oracle)
    print("Evaluating initial population...")
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        if sae_manager:
            sae_manager.oracle.add_training_data(ind, fit[0])

    nevals = len(population)
    
    if sae_manager:
        sae_manager.oracle.train()

    if halloffame:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=nevals, **record)
    print(logbook.stream)

    # Begin the evolution
    for gen in range(1, ngen + 1):
        # 1. Generate Offspring (Crossover and Mutation)
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.8, mutpb=0.3)

        if sae_manager:
            # 2. Screen and Evaluate using SAE Manager
            nevals = sae_manager.screen_and_evaluate(offspring, toolbox)
        else:
            # 2. Standard Evaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            nevals = len(invalid_ind)

        # 3. Update Hall of Fame
        if halloffame:
            halloffame.update(offspring)

        # 4. Selection (Replace the population by the offspring)
        population[:] = toolbox.select(offspring, k=len(population))

        # 5. Record Statistics
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        print(logbook.stream)

    return population, logbook

if __name__ == "__main__":
    # Set run_chimera=True to run the full cycle, False to reuse existing DSG
    main(historical_data_path=None, run_chimera=True, use_sae=True)