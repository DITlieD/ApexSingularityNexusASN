import json
from deap import gp

def _recursive_builder(tree, index):
    """
    A recursive helper function to build the structure for a single DEAP tree.
    """
    node_primitive = tree[index]

    if isinstance(node_primitive, gp.Terminal):
        # Handle features (ARGx)
        if hasattr(node_primitive, 'name') and node_primitive.name.startswith("ARG"):
            primitive = {"Feature": int(node_primitive.name[3:])}
        # Handle constants (Ephemeral or explicit)
        elif hasattr(node_primitive, 'value'):
             # Ensure the value is a standard float for JSON serialization
            primitive = {"Constant": float(node_primitive.value)}
        else:
            # Fallback for unknown terminals
            return None, index + 1
        return {"primitive": primitive, "children": []}, index + 1
    else: # It's a Primitive (Operator)
        children = []
        current_index = index + 1
        for _ in range(node_primitive.arity):
            child_node, next_index = _recursive_builder(tree, current_index)
            if child_node:
                children.append(child_node)
            current_index = next_index

        return {"primitive": node_primitive.name, "children": children}, current_index

def deap_to_json(individual, indent=4):
    """
    Converts a DEAP GP individual (list of trees) into a JSON string for the Rust interpreter.
    """
    # Ensure the individual is a list containing at least two trees
    if not isinstance(individual, list) or len(individual) < 2:
         raise ValueError("Individual must be a list containing 2 trees (Signal and Size).")

    signal_tree = individual[0]
    size_tree = individual[1]

    signal_root, _ = _recursive_builder(signal_tree, 0)
    size_root, _ = _recursive_builder(size_tree, 0)

    if not signal_root or not size_root:
        raise ValueError("Failed to build valid tree structure.")

    # The Rust interpreter (GPStrategy) expects this specific structure
    output = {
        "signal_root": signal_root,
        "size_root": size_root
    }
    return json.dumps(output, indent=indent)

# ... (Example Usage omitted) ...import json
from deap import gp

def _recursive_builder(tree, index):
    """
    A recursive helper function to build the structure for a single DEAP tree.
    """
    node_primitive = tree[index]

    if isinstance(node_primitive, gp.Terminal):
        # Handle features (ARGx)
        if hasattr(node_primitive, 'name') and node_primitive.name.startswith("ARG"):
            primitive = {"Feature": int(node_primitive.name[3:])}
        # Handle constants (Ephemeral or explicit)
        elif hasattr(node_primitive, 'value'):
             # Ensure the value is a standard float for JSON serialization
            primitive = {"Constant": float(node_primitive.value)}
        else:
            # Fallback for unknown terminals
            return None, index + 1
        return {"primitive": primitive, "children": []}, index + 1
    else: # It's a Primitive (Operator)
        children = []
        current_index = index + 1
        for _ in range(node_primitive.arity):
            child_node, next_index = _recursive_builder(tree, current_index)
            if child_node:
                children.append(child_node)
            current_index = next_index

        return {"primitive": node_primitive.name, "children": children}, current_index

def deap_to_json(individual, indent=4):
    """
    Converts a DEAP GP individual (list of trees) into a JSON string for the Rust interpreter.
    """
    # Ensure the individual is a list containing at least two trees
    if not isinstance(individual, list) or len(individual) < 2:
         raise ValueError("Individual must be a list containing 2 trees (Signal and Size).")

    signal_tree = individual[0]
    size_tree = individual[1]

    signal_root, _ = _recursive_builder(signal_tree, 0)
    size_root, _ = _recursive_builder(size_tree, 0)

    if not signal_root or not size_root:
        raise ValueError("Failed to build valid tree structure.")

    # The Rust interpreter (GPStrategy) expects this specific structure
    output = {
        "signal_root": signal_root,
        "size_root": size_root
    }
    return json.dumps(output, indent=indent)

# ... (Example Usage omitted) ...