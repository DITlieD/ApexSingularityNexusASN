import numpy as np
from deap import gp
import lightgbm as lgb
import operator
import logging

logger = logging.getLogger("SAE")
logging.basicConfig(level=logging.INFO)

# --- 1. DNA Vectorizer ---
class DNAVectorizer:
    """
    Converts a dual-tree GP individual into a fixed-size numerical vector.
    """
    def __init__(self, pset: gp.PrimitiveSet):
        self.pset = pset
        self.feature_names = [arg.name for arg in pset.arguments]
        # Extract operator names dynamically from the PSet
        self.operators = []
        for prim_list in pset.primitives.values():
            for prim in prim_list:
                self.operators.append(prim.name)
        self.operators = sorted(list(set(self.operators)))

    def vectorize(self, individual):
        """Vectorizes the signal tree (index 0) and size tree (index 1)."""
        vec_signal = self._vectorize_tree(individual[0])
        vec_size = self._vectorize_tree(individual[1])
        return np.concatenate([vec_signal, vec_size])

    def _vectorize_tree(self, tree: gp.PrimitiveTree):
        features = []
        
        # 1. Structural Features
        tree_size = len(tree)
        tree_depth = tree.height
        features.extend([tree_size, tree_depth])

        # 2. Node Type Distribution (Normalized)
        node_counts = {op: 0 for op in self.operators}
        feature_counts = {f: 0 for f in self.feature_names}
        constant_count = 0

        for node in tree:
            if isinstance(node, gp.Primitive):
                if node.name in node_counts:
                    node_counts[node.name] += 1
            elif isinstance(node, gp.Terminal):
                if hasattr(node, 'name') and node.name in self.feature_names:
                    feature_counts[node.name] += 1
                else:
                    constant_count += 1
        
        # Normalize counts by tree size
        if tree_size > 0:
            op_vec = [node_counts.get(op, 0) / tree_size for op in self.operators]
            feat_vec = [feature_counts.get(f, 0) / tree_size for f in self.feature_names]
            const_norm = constant_count / tree_size
        else:
            op_vec = [0.0] * len(self.operators)
            feat_vec = [0.0] * len(self.feature_names)
            const_norm = 0.0

        features.extend(op_vec)
        features.extend(feat_vec)
        features.append(const_norm)

        # 3. Complexity/Balance Metrics
        op_count = sum(node_counts.values())
        terminal_count = sum(feature_counts.values()) + constant_count
        
        op_terminal_ratio = op_count / (terminal_count + 1e-6)
        balance_metric = tree_depth / (tree_size + 1e-6)
        
        features.extend([op_terminal_ratio, balance_metric])

        return np.array(features)

# --- 2. Fitness Oracle (LightGBM) ---
class FitnessOracle:
    """
    A LightGBM surrogate model that predicts fitness from DNA vectors.
    """
    def __init__(self, vectorizer: DNAVectorizer):
        self.vectorizer = vectorizer
        self.model = None
        self.training_data = {'X': [], 'y': []}
        self.min_samples_to_train = 100

        # Optimized LightGBM parameters
        self.params = {
            'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 150,
            'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1,
        }

    def add_training_data(self, individual, fitness):
        vector = self.vectorizer.vectorize(individual)
        self.training_data['X'].append(vector)
        self.training_data['y'].append(fitness)

    def train(self):
        if len(self.training_data['y']) < self.min_samples_to_train:
            return

        X = np.array(self.training_data['X'])
        y = np.array(self.training_data['y'])

        logger.info(f"[Oracle] Training on {len(y)} samples...")
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)

    def predict_batch(self, individuals):
        if self.model is None:
            # If model is not trained, return random predictions (exploration)
            return np.random.rand(len(individuals)) * -10.0
        
        vectors = [self.vectorizer.vectorize(ind) for ind in individuals]
        return self.model.predict(np.array(vectors))

# --- 3. SAE Controller ---
class SurrogateAssistedEvolution:
    """
    Manages the integration of the Oracle into the evolutionary loop.
    """
    def __init__(self, oracle: FitnessOracle, top_k_percent=0.05):
        self.oracle = oracle
        self.top_k_percent = top_k_percent
        self.generation_count = 0

    def screen_and_evaluate(self, population, toolbox):
        """
        Screens the population using the Oracle and evaluates only the top candidates.
        """
        self.generation_count += 1

        # 1. Predict fitness
        predictions = self.oracle.predict_batch(population)

        # 2. Select the Top K%
        top_k = max(1, int(len(population) * self.top_k_percent))
        indices = np.argsort(predictions)[::-1]
        top_indices = indices[:top_k]

        logger.info(f"[SAE Gen {self.generation_count}] Screening {len(population)}, evaluating top {top_k}...")

        # 3. Evaluate the top candidates using the real fitness function (Rust)
        evaluated_count = 0
        for i in top_indices:
            ind = population[i]
            if not ind.fitness.valid:
                fitness = toolbox.evaluate(ind)
                ind.fitness.values = fitness
                self.oracle.add_training_data(ind, fitness[0])
                evaluated_count += 1

        # 4. Assign worst fitness to unevaluated individuals
        for i in range(len(population)):
            if not population[i].fitness.valid:
                population[i].fitness.values = (-float('inf'),)
        
        # 5. Retrain the Oracle periodically
        if evaluated_count > 0 and self.generation_count % 5 == 0:
             self.oracle.train()

        return evaluated_count