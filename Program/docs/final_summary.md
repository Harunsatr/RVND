# MFVRPTW Final Summary

## Problem Overview
- Scenario: Multi-Fleet Vehicle Routing Problem with Time Windows (MFVRPTW) for a pharmaceutical distribution depot serving hospitals, clinics, and pharmacies.
- Objective: Minimize combined distance, travel/service time, and soft time-window violations while respecting heterogeneous fleet capacities and availability.
- Input scope: 1 depot, 10 customers, 3 vehicle types (A, B, C) with DOCX-defined capacities, costs, and time windows.

## Solution Pipeline
1. Distance/Time Matrix: Euclidean distances derived from coordinates (1 km = 1 minute).
2. Sweep Algorithm: Polar-angle sorting with capacity-based clustering; one vehicle per cluster.
3. Nearest Neighbor (NN): Initial route construction per cluster from depot.
4. Ant Colony System (ACS): Per-cluster optimization (m=2, α=1, β=2, ρ=0.2, q₀=0.85, 2 iterations).
5. RVND: Route refinement via randomized variable neighborhood descent (2-opt, swap, relocate).

## Global Metrics
- Total distance: 272.636 km
- Total time component: 449.636 minutes
- Total TW violation: 0.000 minutes
- Total objective: 722.271
- Total cost: Rp 512636
- Fleet usage: A:1, B:2, C:1

## Routes per Cluster
| Cluster | Vehicle | Sequence | Distance (km) | Time Component (min) | TW Violation (min) | Objective |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | C | 0 → 6 → 10 → 8 → 4 → 0 | 90.270 | 166.270 | 0.000 | 256.540 |
| 2 | B | 0 → 2 → 5 → 0 | 66.831 | 109.831 | 0.000 | 176.661 |
| 3 | B | 0 → 1 → 7 → 0 | 60.043 | 97.043 | 0.000 | 157.085 |
| 4 | A | 0 → 9 → 3 → 0 | 55.492 | 76.492 | 0.000 | 131.984 |

## ACS vs. ACS + RVND Comparison
| Cluster | ACS Distance | RVND Distance | Δ Distance | ACS Objective | RVND Objective | Δ Objective |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 91.124 | 90.270 | -0.854 | 258.248 | 256.540 | -1.708 |
| 2 | 66.831 | 66.831 | 0.000 | 176.661 | 176.661 | 0.000 |
| 3 | 60.043 | 60.043 | 0.000 | 157.085 | 157.085 | 0.000 |
| 4 | 55.492 | 55.492 | 0.000 | 131.984 | 131.984 | 0.000 |
| **Total** | **273.490** | **272.636** | **-0.854** | **723.979** | **722.271** | **-1.708** |

## Validation Checks
- nodes_match: PASS
- distance_symmetric: PASS
- distance_zero_diagonal: PASS
- capacity_respected: PASS
- tw_no_violation: PASS
