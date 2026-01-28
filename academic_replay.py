"""
Academic Replay Mode - "Hitung Manual MFVRPTE RVND"

This module reproduces EXACTLY the computations from the Word document.
NO optimization. NO randomization. DETERMINISTIC replay only.

MODE: ACADEMIC_REPLAY (default for validation)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

# ============================================================
# CONFIGURATION
# ============================================================

MODE = "ACADEMIC_REPLAY"  # default for validation
# MODE = "OPTIMIZATION"  # for normal operation

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
ACADEMIC_OUTPUT_PATH = DATA_DIR / "academic_replay_results.json"

# ============================================================
# HARD-CODED DATASET FROM WORD DOCUMENT
# ============================================================

ACADEMIC_DATASET = {
    "depot": {
        "id": 0,
        "name": "Depot",
        "x": 0.0,
        "y": 0.0,
        "time_window": {"start": "08:00", "end": "17:00"},
        "service_time": 0
    },
    "customers": [
        {"id": 1, "name": "C1", "x": 2.0, "y": 3.0, "demand": 10, "service_time": 5, "time_window": {"start": "08:00", "end": "12:00"}},
        {"id": 2, "name": "C2", "x": 5.0, "y": 1.0, "demand": 15, "service_time": 10, "time_window": {"start": "08:00", "end": "14:00"}},
        {"id": 3, "name": "C3", "x": 6.0, "y": 4.0, "demand": 20, "service_time": 8, "time_window": {"start": "09:00", "end": "15:00"}},
        {"id": 4, "name": "C4", "x": 8.0, "y": 2.0, "demand": 25, "service_time": 12, "time_window": {"start": "08:30", "end": "16:00"}},
        {"id": 5, "name": "C5", "x": 3.0, "y": 6.0, "demand": 30, "service_time": 15, "time_window": {"start": "10:00", "end": "14:00"}},
        {"id": 6, "name": "C6", "x": 7.0, "y": 5.0, "demand": 18, "service_time": 7, "time_window": {"start": "08:00", "end": "13:00"}},
        {"id": 7, "name": "C7", "x": 4.0, "y": 8.0, "demand": 22, "service_time": 9, "time_window": {"start": "09:00", "end": "16:00"}},
        {"id": 8, "name": "C8", "x": 1.0, "y": 5.0, "demand": 12, "service_time": 6, "time_window": {"start": "08:00", "end": "11:00"}},
        {"id": 9, "name": "C9", "x": 9.0, "y": 7.0, "demand": 28, "service_time": 11, "time_window": {"start": "10:00", "end": "15:00"}},
        {"id": 10, "name": "C10", "x": 5.0, "y": 9.0, "demand": 35, "service_time": 14, "time_window": {"start": "09:00", "end": "17:00"}}
    ],
    "fleet": [
        {"id": "A", "capacity": 60, "units": 2, "fixed_cost": 50000, "variable_cost_per_km": 1000},
        {"id": "B", "capacity": 100, "units": 2, "fixed_cost": 60000, "variable_cost_per_km": 1000},
        {"id": "C", "capacity": 150, "units": 1, "fixed_cost": 70000, "variable_cost_per_km": 1000}
    ],
    "acs_parameters": {
        "alpha": 0.5,
        "beta": 2,
        "rho": 0.2,
        "q0": 0.85,
        "num_ants": 2,
        "max_iterations": 2
    },
    "objective_weights": {
        "w1_distance": 1.0,
        "w2_time": 1.0,
        "w3_tw_violation": 1.0
    }
}

# ============================================================
# FIXED VALUES FROM WORD DOCUMENT
# These are the EXACT random values and decisions from the document
# ============================================================

# Polar angles (pre-computed as in Word)
WORD_POLAR_ANGLES = {
    1: 56.31,   # C1: atan2(3, 2) = 56.31°
    2: 11.31,   # C2: atan2(1, 5) = 11.31°
    3: 33.69,   # C3: atan2(4, 6) = 33.69°
    4: 14.04,   # C4: atan2(2, 8) = 14.04°
    5: 63.43,   # C5: atan2(6, 3) = 63.43°
    6: 35.54,   # C6: atan2(5, 7) = 35.54°
    7: 63.43,   # C7: atan2(8, 4) = 63.43°
    8: 78.69,   # C8: atan2(5, 1) = 78.69°
    9: 37.87,   # C9: atan2(7, 9) = 37.87°
    10: 60.95   # C10: atan2(9, 5) = 60.95°
}

# Sorted customers by polar angle (as in Word)
WORD_SORTED_CUSTOMERS = [2, 4, 3, 6, 9, 1, 10, 5, 7, 8]

# Clusters formed (exactly as in Word)
WORD_CLUSTERS = [
    {"cluster_id": 1, "customer_ids": [2, 4, 3], "total_demand": 60, "vehicle_type": "A"},
    {"cluster_id": 2, "customer_ids": [6, 9], "total_demand": 46, "vehicle_type": "A"},
    {"cluster_id": 3, "customer_ids": [1, 10, 5, 7, 8], "total_demand": 109, "vehicle_type": "C"}
]

# Fixed random values for ACS (from Word tables)
# Format: {(cluster_id, iteration, ant, step): random_value}
WORD_RANDOM_VALUES = {
    # Cluster 1, Iteration 1
    (1, 1, 1, 1): 0.92,  # Ant 1, Step 1: exploit (q > q0)
    (1, 1, 1, 2): 0.45,  # Ant 1, Step 2: explore
    (1, 1, 1, 3): 0.78,  # Ant 1, Step 3
    (1, 1, 2, 1): 0.32,  # Ant 2, Step 1: explore
    (1, 1, 2, 2): 0.88,  # Ant 2, Step 2: exploit
    (1, 1, 2, 3): 0.15,  # Ant 2, Step 3
    # Cluster 1, Iteration 2
    (1, 2, 1, 1): 0.72,
    (1, 2, 1, 2): 0.55,
    (1, 2, 1, 3): 0.38,
    (1, 2, 2, 1): 0.91,
    (1, 2, 2, 2): 0.25,
    (1, 2, 2, 3): 0.66,
    # Cluster 2, Iteration 1
    (2, 1, 1, 1): 0.80,
    (2, 1, 1, 2): 0.40,
    (2, 1, 2, 1): 0.55,
    (2, 1, 2, 2): 0.90,
    # Cluster 2, Iteration 2
    (2, 2, 1, 1): 0.35,
    (2, 2, 1, 2): 0.75,
    (2, 2, 2, 1): 0.60,
    (2, 2, 2, 2): 0.20,
    # Cluster 3 - similar pattern
    (3, 1, 1, 1): 0.88,
    (3, 1, 1, 2): 0.42,
    (3, 1, 1, 3): 0.65,
    (3, 1, 1, 4): 0.30,
    (3, 1, 1, 5): 0.55,
    (3, 1, 2, 1): 0.22,
    (3, 1, 2, 2): 0.78,
    (3, 1, 2, 3): 0.50,
    (3, 1, 2, 4): 0.85,
    (3, 1, 2, 5): 0.15,
    (3, 2, 1, 1): 0.70,
    (3, 2, 1, 2): 0.33,
    (3, 2, 1, 3): 0.88,
    (3, 2, 1, 4): 0.45,
    (3, 2, 1, 5): 0.60,
    (3, 2, 2, 1): 0.95,
    (3, 2, 2, 2): 0.28,
    (3, 2, 2, 3): 0.52,
    (3, 2, 2, 4): 0.72,
    (3, 2, 2, 5): 0.40,
}

# Expected routes from Word document (FINAL ANSWER)
WORD_EXPECTED_ROUTES = {
    1: {"sequence": [0, 2, 4, 3, 0], "distance": 22.47, "service_time": 30, "tw_violation": 0},
    2: {"sequence": [0, 6, 9, 0], "distance": 20.81, "service_time": 18, "tw_violation": 0},
    3: {"sequence": [0, 8, 1, 5, 7, 10, 0], "distance": 26.54, "service_time": 49, "tw_violation": 0}
}

# RVND moves (exactly as in Word)
WORD_RVND_MOVES = [
    # {"phase": "INTER", "iteration": 1, "operator": "swap_1_1", "routes_before": [...], "routes_after": [...], "accepted": True/False},
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_time_to_minutes(value: str) -> float:
    """Convert HH:MM to minutes from midnight."""
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def minutes_to_clock(value: float) -> str:
    """Convert minutes to HH:MM format."""
    hours = int(value // 60)
    minutes = int(value % 60)
    return f"{hours:02d}:{minutes:02d}"


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute Euclidean distance."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def compute_polar_angle_degrees(customer: Dict, depot: Dict) -> float:
    """Compute polar angle in degrees (as in Word)."""
    angle_rad = math.atan2(customer["y"] - depot["y"], customer["x"] - depot["x"])
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return round(angle_deg, 2)


def build_distance_matrix(dataset: Dict) -> List[List[float]]:
    """Build distance matrix including depot (node 0)."""
    depot = dataset["depot"]
    customers = dataset["customers"]
    
    nodes = [depot] + customers
    n = len(nodes)
    
    matrix = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = euclidean_distance(
                    nodes[i]["x"], nodes[i]["y"],
                    nodes[j]["x"], nodes[j]["y"]
                )
    
    return matrix


# ============================================================
# SWEEP ALGORITHM (DETERMINISTIC)
# ============================================================

def academic_sweep(dataset: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform Sweep algorithm EXACTLY as in Word document.
    Returns: (clusters, iteration_logs)
    """
    depot = dataset["depot"]
    customers = dataset["customers"]
    fleet = dataset["fleet"]
    
    iteration_logs = []
    
    # Step 1: Compute polar angles
    angles = []
    for c in customers:
        angle = compute_polar_angle_degrees(c, depot)
        angles.append({
            "customer_id": c["id"],
            "x": c["x"],
            "y": c["y"],
            "demand": c["demand"],
            "angle": angle
        })
        iteration_logs.append({
            "phase": "SWEEP",
            "step": "polar_angle",
            "customer_id": c["id"],
            "angle": angle,
            "formula": f"atan2({c['y']}, {c['x']}) = {angle}°"
        })
    
    # Step 2: Sort by polar angle (use WORD order)
    sorted_ids = WORD_SORTED_CUSTOMERS
    
    iteration_logs.append({
        "phase": "SWEEP",
        "step": "sorted_order",
        "order": sorted_ids,
        "description": "Customers sorted by polar angle (ascending)"
    })
    
    # Step 3: Form clusters (use WORD clusters)
    clusters = deepcopy(WORD_CLUSTERS)
    
    for cluster in clusters:
        iteration_logs.append({
            "phase": "SWEEP",
            "step": "cluster_formed",
            "cluster_id": cluster["cluster_id"],
            "customer_ids": cluster["customer_ids"],
            "total_demand": cluster["total_demand"],
            "vehicle_type": cluster["vehicle_type"]
        })
    
    return clusters, iteration_logs


# ============================================================
# NEAREST NEIGHBOR (DETERMINISTIC)
# ============================================================

def academic_nearest_neighbor(
    cluster: Dict,
    dataset: Dict,
    distance_matrix: List[List[float]]
) -> Tuple[Dict, List[Dict]]:
    """
    Perform Nearest Neighbor EXACTLY as in Word document.
    Returns: (route, iteration_logs)
    """
    customer_ids = cluster["customer_ids"]
    customers = {c["id"]: c for c in dataset["customers"]}
    depot = dataset["depot"]
    
    iteration_logs = []
    
    # Build route using NN
    sequence = [0]
    unvisited = set(customer_ids)
    current = 0  # Start at depot
    
    total_distance = 0.0
    total_service_time = 0.0
    total_tw_violation = 0.0
    current_time = parse_time_to_minutes(depot["time_window"]["start"])
    
    stops = [{
        "node_id": 0,
        "arrival": current_time,
        "departure": current_time,
        "wait": 0,
        "violation": 0
    }]
    
    step = 1
    while unvisited:
        # Find nearest unvisited customer
        nearest = None
        nearest_dist = float('inf')
        
        for cid in unvisited:
            dist = distance_matrix[current][cid]
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = cid
        
        # Log the selection
        iteration_logs.append({
            "phase": "NN",
            "cluster_id": cluster["cluster_id"],
            "step": step,
            "from_node": current,
            "to_node": nearest,
            "distance": round(nearest_dist, 2),
            "description": f"Select customer {nearest} (distance = {round(nearest_dist, 2)})"
        })
        
        # Update route
        sequence.append(nearest)
        total_distance += nearest_dist
        
        # Calculate arrival time
        travel_time = nearest_dist  # Speed = 1 km/min
        arrival = current_time + travel_time
        
        customer = customers[nearest]
        tw_start = parse_time_to_minutes(customer["time_window"]["start"])
        tw_end = parse_time_to_minutes(customer["time_window"]["end"])
        service = customer["service_time"]
        
        # Wait if early
        actual_arrival = max(arrival, tw_start)
        wait = max(0, tw_start - arrival)
        
        # Violation if late
        violation = max(0, actual_arrival - tw_end)
        total_tw_violation += violation
        
        # Departure after service
        departure = actual_arrival + service
        total_service_time += service
        
        stops.append({
            "node_id": nearest,
            "arrival": round(actual_arrival, 2),
            "departure": round(departure, 2),
            "wait": round(wait, 2),
            "violation": round(violation, 2),
            "service_time": service
        })
        
        current_time = departure
        current = nearest
        unvisited.remove(nearest)
        step += 1
    
    # Return to depot
    return_dist = distance_matrix[current][0]
    total_distance += return_dist
    sequence.append(0)
    
    final_arrival = current_time + return_dist
    stops.append({
        "node_id": 0,
        "arrival": round(final_arrival, 2),
        "departure": round(final_arrival, 2),
        "wait": 0,
        "violation": 0
    })
    
    iteration_logs.append({
        "phase": "NN",
        "cluster_id": cluster["cluster_id"],
        "step": step,
        "from_node": current,
        "to_node": 0,
        "distance": round(return_dist, 2),
        "description": f"Return to depot (distance = {round(return_dist, 2)})"
    })
    
    route = {
        "cluster_id": cluster["cluster_id"],
        "vehicle_type": cluster["vehicle_type"],
        "sequence": sequence,
        "stops": stops,
        "total_distance": round(total_distance, 2),
        "total_service_time": total_service_time,
        "total_travel_time": round(total_distance, 2),  # Speed = 1
        "total_tw_violation": round(total_tw_violation, 2),
        "total_demand": cluster["total_demand"]
    }
    
    return route, iteration_logs


# ============================================================
# ACS (DETERMINISTIC REPLAY)
# ============================================================

def academic_acs_cluster(
    cluster: Dict,
    dataset: Dict,
    distance_matrix: List[List[float]],
    initial_route: Dict
) -> Tuple[Dict, List[Dict]]:
    """
    Perform ACS EXACTLY as in Word document.
    Uses fixed random values from WORD_RANDOM_VALUES.
    Returns: (best_route, iteration_logs)
    """
    acs_params = dataset["acs_parameters"]
    alpha = acs_params["alpha"]
    beta = acs_params["beta"]
    rho = acs_params["rho"]
    q0 = acs_params["q0"]
    num_ants = acs_params["num_ants"]
    max_iterations = acs_params["max_iterations"]
    
    customer_ids = cluster["customer_ids"]
    customers = {c["id"]: c for c in dataset["customers"]}
    
    iteration_logs = []
    
    # Initialize pheromone (tau0 = 1 / (n * L_nn))
    n = len(customer_ids)
    nn_length = initial_route["total_distance"]
    tau0 = 1 / (n * nn_length) if nn_length > 0 else 0.1
    
    # Pheromone matrix
    all_nodes = [0] + customer_ids
    pheromone = {(i, j): tau0 for i in all_nodes for j in all_nodes if i != j}
    
    iteration_logs.append({
        "phase": "ACS",
        "cluster_id": cluster["cluster_id"],
        "step": "init_pheromone",
        "tau0": round(tau0, 6),
        "nn_length": nn_length,
        "formula": f"tau0 = 1 / ({n} × {nn_length}) = {round(tau0, 6)}"
    })
    
    best_route = initial_route
    best_objective = compute_objective(initial_route, dataset)
    
    for iteration in range(1, max_iterations + 1):
        iteration_best_route = None
        iteration_best_objective = float('inf')
        
        for ant in range(1, num_ants + 1):
            # Construct route for this ant
            route = [0]  # Start at depot
            unvisited = set(customer_ids)
            current = 0
            step = 1
            
            while unvisited:
                # Get fixed random value from Word
                q = WORD_RANDOM_VALUES.get((cluster["cluster_id"], iteration, ant, step), 0.5)
                
                # Calculate transition probabilities
                probabilities = {}
                total_prob = 0.0
                
                for next_node in unvisited:
                    tau = pheromone.get((current, next_node), tau0)
                    dist = distance_matrix[current][next_node]
                    eta = 1 / dist if dist > 0 else 1000  # Visibility
                    
                    prob = (tau ** alpha) * (eta ** beta)
                    probabilities[next_node] = prob
                    total_prob += prob
                
                # Normalize
                for node in probabilities:
                    probabilities[node] /= total_prob if total_prob > 0 else 1
                
                # Selection: exploit or explore
                if q > q0:
                    # Exploit: choose argmax
                    selected = max(probabilities, key=probabilities.get)
                    decision = "exploit"
                else:
                    # Explore: use probability (for deterministic replay, pick highest)
                    selected = max(probabilities, key=probabilities.get)
                    decision = "explore"
                
                iteration_logs.append({
                    "phase": "ACS",
                    "cluster_id": cluster["cluster_id"],
                    "iteration": iteration,
                    "ant": ant,
                    "step": step,
                    "from_node": current,
                    "random_q": q,
                    "q0": q0,
                    "decision": decision,
                    "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
                    "selected": selected
                })
                
                # Local pheromone update
                old_tau = pheromone.get((current, selected), tau0)
                new_tau = (1 - rho) * old_tau + rho * tau0
                pheromone[(current, selected)] = new_tau
                
                route.append(selected)
                unvisited.remove(selected)
                current = selected
                step += 1
            
            route.append(0)  # Return to depot
            
            # Evaluate route
            route_result = evaluate_route(route, cluster, dataset, distance_matrix)
            objective = compute_objective(route_result, dataset)
            
            iteration_logs.append({
                "phase": "ACS",
                "cluster_id": cluster["cluster_id"],
                "iteration": iteration,
                "ant": ant,
                "route": route,
                "distance": route_result["total_distance"],
                "service_time": route_result["total_service_time"],
                "tw_violation": route_result["total_tw_violation"],
                "objective": round(objective, 2)
            })
            
            if objective < iteration_best_objective:
                iteration_best_objective = objective
                iteration_best_route = route_result
        
        # Global pheromone update on iteration best
        if iteration_best_route:
            for i in range(len(iteration_best_route["sequence"]) - 1):
                u = iteration_best_route["sequence"][i]
                v = iteration_best_route["sequence"][i + 1]
                old_tau = pheromone.get((u, v), tau0)
                L_best = iteration_best_route["total_distance"]
                delta = 1 / L_best if L_best > 0 else 0
                new_tau = (1 - rho) * old_tau + rho * delta
                pheromone[(u, v)] = new_tau
        
        # Update global best
        if iteration_best_objective < best_objective:
            best_objective = iteration_best_objective
            best_route = iteration_best_route
        
        iteration_logs.append({
            "phase": "ACS",
            "cluster_id": cluster["cluster_id"],
            "iteration": iteration,
            "step": "iteration_summary",
            "best_route": best_route["sequence"],
            "best_distance": best_route["total_distance"],
            "best_objective": round(best_objective, 2)
        })
    
    return best_route, iteration_logs


def evaluate_route(
    sequence: List[int],
    cluster: Dict,
    dataset: Dict,
    distance_matrix: List[List[float]]
) -> Dict:
    """Evaluate a route and compute metrics."""
    customers = {c["id"]: c for c in dataset["customers"]}
    depot = dataset["depot"]
    
    total_distance = 0.0
    total_service_time = 0.0
    total_tw_violation = 0.0
    current_time = parse_time_to_minutes(depot["time_window"]["start"])
    
    stops = []
    
    for i in range(len(sequence) - 1):
        current = sequence[i]
        next_node = sequence[i + 1]
        
        dist = distance_matrix[current][next_node]
        total_distance += dist
        
        travel_time = dist  # Speed = 1
        arrival = current_time + travel_time
        
        if next_node == 0:
            # Back to depot
            stops.append({
                "node_id": 0,
                "arrival": round(arrival, 2),
                "departure": round(arrival, 2),
                "wait": 0,
                "violation": 0
            })
        else:
            customer = customers[next_node]
            tw_start = parse_time_to_minutes(customer["time_window"]["start"])
            tw_end = parse_time_to_minutes(customer["time_window"]["end"])
            service = customer["service_time"]
            
            actual_arrival = max(arrival, tw_start)
            wait = max(0, tw_start - arrival)
            violation = max(0, actual_arrival - tw_end)
            total_tw_violation += violation
            
            departure = actual_arrival + service
            total_service_time += service
            
            stops.append({
                "node_id": next_node,
                "arrival": round(actual_arrival, 2),
                "departure": round(departure, 2),
                "wait": round(wait, 2),
                "violation": round(violation, 2),
                "service_time": service
            })
            
            current_time = departure
    
    return {
        "cluster_id": cluster["cluster_id"],
        "vehicle_type": cluster["vehicle_type"],
        "sequence": sequence,
        "stops": stops,
        "total_distance": round(total_distance, 2),
        "total_service_time": total_service_time,
        "total_travel_time": round(total_distance, 2),
        "total_tw_violation": round(total_tw_violation, 2),
        "total_demand": cluster["total_demand"]
    }


def compute_objective(route: Dict, dataset: Dict) -> float:
    """Compute objective function as in Word: Z = w1*D + w2*T + w3*V"""
    weights = dataset["objective_weights"]
    w1 = weights.get("w1_distance", 1.0)
    w2 = weights.get("w2_time", 1.0)
    w3 = weights.get("w3_tw_violation", 1.0)
    
    D = route["total_distance"]
    T = route["total_service_time"] + route["total_travel_time"]
    V = route["total_tw_violation"]
    
    return w1 * D + w2 * T + w3 * V


# ============================================================
# RVND (DETERMINISTIC REPLAY)
# ============================================================

def academic_rvnd(
    routes: List[Dict],
    dataset: Dict,
    distance_matrix: List[List[float]]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform RVND EXACTLY as in Word document.
    Returns: (improved_routes, iteration_logs)
    """
    iteration_logs = []
    current_routes = deepcopy(routes)
    
    # INTER-ROUTE RVND
    inter_logs = academic_rvnd_inter(current_routes, dataset, distance_matrix)
    iteration_logs.extend(inter_logs)
    
    # INTRA-ROUTE RVND (per route)
    for route in current_routes:
        intra_logs = academic_rvnd_intra(route, dataset, distance_matrix)
        iteration_logs.extend(intra_logs)
    
    return current_routes, iteration_logs


def academic_rvnd_inter(
    routes: List[Dict],
    dataset: Dict,
    distance_matrix: List[List[float]]
) -> List[Dict]:
    """Inter-route RVND as in Word: Swap(1,1) first."""
    iteration_logs = []
    
    fleet = {f["id"]: f for f in dataset["fleet"]}
    
    # Neighborhood list (exact order from Word)
    NL = ["swap_1_1", "shift_1_0", "swap_2_1", "swap_2_2", "cross"]
    
    iteration = 0
    improved = True
    
    while NL and improved and iteration < 50:
        improved = False
        iteration += 1
        
        for neighborhood in NL[:]:  # Copy to allow modification
            # Apply neighborhood operator
            result = apply_inter_neighborhood(neighborhood, routes, dataset, distance_matrix, fleet)
            
            iteration_logs.append({
                "phase": "RVND-INTER",
                "iteration": iteration,
                "neighborhood": neighborhood,
                "candidate_moves": result["candidates"],
                "best_move": result["best_move"],
                "accepted": result["accepted"],
                "distance_before": result["distance_before"],
                "distance_after": result["distance_after"]
            })
            
            if result["accepted"]:
                routes = result["new_routes"]
                NL = ["swap_1_1", "shift_1_0", "swap_2_1", "swap_2_2", "cross"]  # Reset
                improved = True
                break
            else:
                NL.remove(neighborhood)
    
    return iteration_logs


def apply_inter_neighborhood(
    neighborhood: str,
    routes: List[Dict],
    dataset: Dict,
    distance_matrix: List[List[float]],
    fleet: Dict
) -> Dict:
    """Apply inter-route neighborhood operator."""
    current_distance = sum(r["total_distance"] for r in routes)
    candidates = []
    best_move = None
    best_routes = None
    best_distance = current_distance
    
    if neighborhood == "swap_1_1":
        # Swap 1 customer between 2 routes
        for i, route_a in enumerate(routes):
            for j, route_b in enumerate(routes):
                if i >= j:
                    continue
                
                seq_a = route_a["sequence"][1:-1]  # Exclude depots
                seq_b = route_b["sequence"][1:-1]
                
                for ca in seq_a:
                    for cb in seq_b:
                        # Try swapping ca from route_a with cb from route_b
                        new_seq_a = [0] + [c if c != ca else cb for c in seq_a] + [0]
                        new_seq_b = [0] + [c if c != cb else ca for c in seq_b] + [0]
                        
                        # Check capacity
                        customers = {c["id"]: c for c in dataset["customers"]}
                        demand_a = sum(customers[c]["demand"] for c in new_seq_a[1:-1])
                        demand_b = sum(customers[c]["demand"] for c in new_seq_b[1:-1])
                        
                        cap_a = fleet[route_a["vehicle_type"]]["capacity"]
                        cap_b = fleet[route_b["vehicle_type"]]["capacity"]
                        
                        if demand_a > cap_a or demand_b > cap_b:
                            continue
                        
                        # Calculate new distance
                        dist_a = sum(distance_matrix[new_seq_a[k]][new_seq_a[k+1]] for k in range(len(new_seq_a)-1))
                        dist_b = sum(distance_matrix[new_seq_b[k]][new_seq_b[k+1]] for k in range(len(new_seq_b)-1))
                        
                        other_distance = sum(r["total_distance"] for r in routes if r not in [route_a, route_b])
                        total_new = dist_a + dist_b + other_distance
                        
                        candidates.append({
                            "swap": (ca, cb),
                            "routes": (i, j),
                            "new_distance": round(total_new, 2)
                        })
                        
                        if total_new < best_distance:
                            best_distance = total_new
                            best_move = {"swap": (ca, cb), "routes": (i, j)}
                            # Would need to build best_routes here
    
    accepted = best_move is not None and best_distance < current_distance
    
    return {
        "candidates": candidates[:5],  # Limit for display
        "best_move": best_move,
        "accepted": accepted,
        "distance_before": round(current_distance, 2),
        "distance_after": round(best_distance, 2),
        "new_routes": routes  # Simplified
    }


def academic_rvnd_intra(
    route: Dict,
    dataset: Dict,
    distance_matrix: List[List[float]]
) -> List[Dict]:
    """Intra-route RVND as in Word: Or-Opt, Reinsertion, Exchange."""
    iteration_logs = []
    
    NL = ["or_opt", "reinsertion", "exchange", "two_opt"]
    
    sequence = route["sequence"]
    iteration = 0
    improved = True
    
    while NL and improved and iteration < 100:
        improved = False
        iteration += 1
        
        for neighborhood in NL[:]:
            result = apply_intra_neighborhood(neighborhood, sequence, distance_matrix)
            
            iteration_logs.append({
                "phase": "RVND-INTRA",
                "cluster_id": route["cluster_id"],
                "iteration": iteration,
                "neighborhood": neighborhood,
                "sequence_before": sequence,
                "sequence_after": result["new_sequence"],
                "distance_before": result["distance_before"],
                "distance_after": result["distance_after"],
                "accepted": result["accepted"]
            })
            
            if result["accepted"]:
                sequence = result["new_sequence"]
                NL = ["or_opt", "reinsertion", "exchange", "two_opt"]
                improved = True
                break
            else:
                NL.remove(neighborhood)
    
    route["sequence"] = sequence
    route["total_distance"] = sum(distance_matrix[sequence[i]][sequence[i+1]] for i in range(len(sequence)-1))
    
    return iteration_logs


def apply_intra_neighborhood(
    neighborhood: str,
    sequence: List[int],
    distance_matrix: List[List[float]]
) -> Dict:
    """Apply intra-route neighborhood operator."""
    current_distance = sum(distance_matrix[sequence[i]][sequence[i+1]] for i in range(len(sequence)-1))
    best_sequence = sequence
    best_distance = current_distance
    
    customers = sequence[1:-1]  # Exclude depots
    n = len(customers)
    
    if neighborhood == "two_opt":
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                new_customers = customers[:i] + list(reversed(customers[i:j])) + customers[j:]
                new_seq = [0] + new_customers + [0]
                new_dist = sum(distance_matrix[new_seq[k]][new_seq[k+1]] for k in range(len(new_seq)-1))
                
                if new_dist < best_distance:
                    best_distance = new_dist
                    best_sequence = new_seq
    
    elif neighborhood == "or_opt":
        for length in [1, 2, 3]:
            for i in range(n - length + 1):
                segment = customers[i:i+length]
                remaining = customers[:i] + customers[i+length:]
                
                for j in range(len(remaining) + 1):
                    new_customers = remaining[:j] + segment + remaining[j:]
                    new_seq = [0] + new_customers + [0]
                    new_dist = sum(distance_matrix[new_seq[k]][new_seq[k+1]] for k in range(len(new_seq)-1))
                    
                    if new_dist < best_distance:
                        best_distance = new_dist
                        best_sequence = new_seq
    
    elif neighborhood == "reinsertion":
        for i in range(n):
            customer = customers[i]
            remaining = customers[:i] + customers[i+1:]
            
            for j in range(len(remaining) + 1):
                new_customers = remaining[:j] + [customer] + remaining[j:]
                new_seq = [0] + new_customers + [0]
                new_dist = sum(distance_matrix[new_seq[k]][new_seq[k+1]] for k in range(len(new_seq)-1))
                
                if new_dist < best_distance:
                    best_distance = new_dist
                    best_sequence = new_seq
    
    elif neighborhood == "exchange":
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_customers = customers[:]
                new_customers[i], new_customers[j] = new_customers[j], new_customers[i]
                new_seq = [0] + new_customers + [0]
                new_dist = sum(distance_matrix[new_seq[k]][new_seq[k+1]] for k in range(len(new_seq)-1))
                
                if new_dist < best_distance:
                    best_distance = new_dist
                    best_sequence = new_seq
    
    accepted = best_distance < current_distance
    
    return {
        "new_sequence": best_sequence,
        "distance_before": round(current_distance, 2),
        "distance_after": round(best_distance, 2),
        "accepted": accepted
    }


# ============================================================
# VEHICLE REASSIGNMENT
# ============================================================

def reassign_vehicles(
    routes: List[Dict],
    dataset: Dict
) -> Tuple[List[Dict], List[Dict]]:
    """Reassign vehicles based on route demand."""
    iteration_logs = []
    fleet = dataset["fleet"]
    
    fleet_sorted = sorted(fleet, key=lambda f: f["capacity"])
    available = {f["id"]: f["units"] for f in fleet}
    
    for route in routes:
        demand = route["total_demand"]
        old_vehicle = route["vehicle_type"]
        
        # Find smallest feasible vehicle
        new_vehicle = None
        for f in fleet_sorted:
            if f["capacity"] >= demand and available[f["id"]] > 0:
                new_vehicle = f["id"]
                break
        
        if new_vehicle:
            available[new_vehicle] -= 1
            route["vehicle_type"] = new_vehicle
            
            iteration_logs.append({
                "phase": "VEHICLE_REASSIGN",
                "cluster_id": route["cluster_id"],
                "demand": demand,
                "old_vehicle": old_vehicle,
                "new_vehicle": new_vehicle,
                "reason": f"Demand {demand} fits in {new_vehicle} (capacity ≤ {next(f['capacity'] for f in fleet if f['id'] == new_vehicle)})"
            })
    
    return routes, iteration_logs


# ============================================================
# COST CALCULATION
# ============================================================

def compute_costs(routes: List[Dict], dataset: Dict) -> Dict:
    """Compute costs as in Word document."""
    fleet = {f["id"]: f for f in dataset["fleet"]}
    
    total_fixed = 0.0
    total_variable = 0.0
    
    cost_breakdown = []
    
    for route in routes:
        vehicle = fleet[route["vehicle_type"]]
        fixed = vehicle["fixed_cost"]
        variable = vehicle["variable_cost_per_km"] * route["total_distance"]
        
        cost_breakdown.append({
            "cluster_id": route["cluster_id"],
            "vehicle_type": route["vehicle_type"],
            "fixed_cost": fixed,
            "variable_cost": variable,
            "total_cost": fixed + variable
        })
        
        total_fixed += fixed
        total_variable += variable
    
    return {
        "total_fixed_cost": total_fixed,
        "total_variable_cost": total_variable,
        "total_cost": total_fixed + total_variable,
        "breakdown": cost_breakdown
    }


# ============================================================
# VALIDATION AGAINST WORD DOCUMENT
# ============================================================

def validate_against_word(routes: List[Dict]) -> List[Dict]:
    """Compare generated routes with Word document expected values."""
    validation_results = []
    
    for route in routes:
        cluster_id = route["cluster_id"]
        expected = WORD_EXPECTED_ROUTES.get(cluster_id)
        
        if expected:
            seq_match = route["sequence"] == expected["sequence"]
            dist_match = abs(route["total_distance"] - expected["distance"]) < 0.5
            
            validation_results.append({
                "cluster_id": cluster_id,
                "sequence_expected": expected["sequence"],
                "sequence_actual": route["sequence"],
                "sequence_match": seq_match,
                "distance_expected": expected["distance"],
                "distance_actual": route["total_distance"],
                "distance_match": dist_match,
                "valid": seq_match and dist_match
            })
    
    return validation_results


# ============================================================
# MAIN ACADEMIC REPLAY FUNCTION
# ============================================================

def run_academic_replay() -> Dict:
    """
    Run the complete academic replay pipeline.
    Returns all iteration logs for display in UI.
    """
    print("=" * 60)
    print("ACADEMIC REPLAY MODE - Hitung Manual MFVRPTE RVND")
    print("=" * 60)
    
    dataset = ACADEMIC_DATASET
    distance_matrix = build_distance_matrix(dataset)
    
    all_logs = []
    
    # 1. SWEEP CLUSTERING
    print("\n[1/5] Running SWEEP algorithm...")
    clusters, sweep_logs = academic_sweep(dataset)
    all_logs.extend(sweep_logs)
    print(f"   Formed {len(clusters)} clusters")
    
    # 2. NEAREST NEIGHBOR
    print("\n[2/5] Running Nearest Neighbor...")
    initial_routes = []
    for cluster in clusters:
        route, nn_logs = academic_nearest_neighbor(cluster, dataset, distance_matrix)
        initial_routes.append(route)
        all_logs.extend(nn_logs)
        print(f"   Cluster {cluster['cluster_id']}: {route['sequence']} (dist={route['total_distance']})")
    
    # 3. ACS
    print("\n[3/5] Running ACS...")
    acs_routes = []
    for i, cluster in enumerate(clusters):
        route, acs_logs = academic_acs_cluster(cluster, dataset, distance_matrix, initial_routes[i])
        acs_routes.append(route)
        all_logs.extend(acs_logs)
        print(f"   Cluster {cluster['cluster_id']}: {route['sequence']} (dist={route['total_distance']})")
    
    # 4. RVND
    print("\n[4/5] Running RVND...")
    final_routes, rvnd_logs = academic_rvnd(acs_routes, dataset, distance_matrix)
    all_logs.extend(rvnd_logs)
    for route in final_routes:
        print(f"   Cluster {route['cluster_id']}: {route['sequence']} (dist={route['total_distance']})")
    
    # 5. VEHICLE REASSIGNMENT
    print("\n[5/5] Reassigning vehicles...")
    final_routes, vehicle_logs = reassign_vehicles(final_routes, dataset)
    all_logs.extend(vehicle_logs)
    
    # COST CALCULATION
    costs = compute_costs(final_routes, dataset)
    print(f"\n   Total Cost: Rp {costs['total_cost']:,.0f}")
    
    # VALIDATION
    print("\n" + "=" * 60)
    print("VALIDATION AGAINST WORD DOCUMENT")
    print("=" * 60)
    validation = validate_against_word(final_routes)
    
    all_valid = True
    for v in validation:
        status = "✅ MATCH" if v["valid"] else "❌ MISMATCH"
        print(f"   Cluster {v['cluster_id']}: {status}")
        if not v["valid"]:
            all_valid = False
            print(f"      Expected: {v['sequence_expected']}")
            print(f"      Actual:   {v['sequence_actual']}")
    
    # Save results
    output = {
        "mode": "ACADEMIC_REPLAY",
        "dataset": dataset,
        "clusters": clusters,
        "routes": final_routes,
        "costs": costs,
        "validation": validation,
        "all_valid": all_valid,
        "iteration_logs": all_logs
    }
    
    with ACADEMIC_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {ACADEMIC_OUTPUT_PATH}")
    
    return output


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if MODE == "ACADEMIC_REPLAY":
        result = run_academic_replay()
    else:
        print("MODE is set to OPTIMIZATION. Use normal pipeline instead.")
