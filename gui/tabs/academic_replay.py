"""
Academic Replay Results Tab

Displays all iterations from the Word document validation:
- ACS iterations (per cluster)
- RVND inter-route iterations
- RVND intra-route iterations
- Final validation results
"""

from __future__ import annotations

import json
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
ACADEMIC_OUTPUT_PATH = DATA_DIR / "academic_replay_results.json"


def _format_number(value: float) -> str:
    """Format number with 2 decimals."""
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _load_academic_results() -> Dict[str, Any]:
    """Load academic replay results if available."""
    if ACADEMIC_OUTPUT_PATH.exists():
        with ACADEMIC_OUTPUT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _display_sweep_iterations(logs: List[Dict]) -> None:
    """Display SWEEP algorithm iterations."""
    st.markdown("### 📐 SWEEP Algorithm - Polar Angles & Clustering")
    
    # Polar angles
    angle_logs = [l for l in logs if l.get("phase") == "SWEEP" and l.get("step") == "polar_angle"]
    if angle_logs:
        st.markdown("**Step 1: Polar Angle Computation**")
        df_angles = pd.DataFrame([{
            "Customer": l["customer_id"],
            "Angle (°)": l["angle"],
            "Formula": l["formula"]
        } for l in angle_logs])
        st.dataframe(df_angles, use_container_width=True, hide_index=True)
    
    # Sorted order
    sorted_logs = [l for l in logs if l.get("phase") == "SWEEP" and l.get("step") == "sorted_order"]
    if sorted_logs:
        st.markdown("**Step 2: Sorted Customer Order**")
        st.info(f"Order: {sorted_logs[0]['order']}")
    
    # Clusters formed
    cluster_logs = [l for l in logs if l.get("phase") == "SWEEP" and l.get("step") == "cluster_formed"]
    if cluster_logs:
        st.markdown("**Step 3: Clusters Formed**")
        df_clusters = pd.DataFrame([{
            "Cluster": l["cluster_id"],
            "Customers": str(l["customer_ids"]),
            "Total Demand": l["total_demand"],
            "Vehicle": l["vehicle_type"]
        } for l in cluster_logs])
        st.dataframe(df_clusters, use_container_width=True, hide_index=True)


def _display_nn_iterations(logs: List[Dict]) -> None:
    """Display Nearest Neighbor iterations."""
    st.markdown("### 🔗 Nearest Neighbor - Initial Routes")
    
    nn_logs = [l for l in logs if l.get("phase") == "NN"]
    
    if nn_logs:
        # Group by cluster
        clusters = set(l["cluster_id"] for l in nn_logs)
        
        for cluster_id in sorted(clusters):
            with st.expander(f"Cluster {cluster_id}", expanded=True):
                cluster_logs = [l for l in nn_logs if l["cluster_id"] == cluster_id]
                df = pd.DataFrame([{
                    "Step": l["step"],
                    "From": l["from_node"],
                    "To": l["to_node"],
                    "Distance": l["distance"],
                    "Description": l["description"]
                } for l in cluster_logs])
                st.dataframe(df, use_container_width=True, hide_index=True)


def _display_acs_iterations(logs: List[Dict]) -> None:
    """Display ACS iterations with full detail."""
    st.markdown("### 🐜 Ant Colony System - Iterations")
    
    acs_logs = [l for l in logs if l.get("phase") == "ACS"]
    
    if not acs_logs:
        st.info("No ACS iteration logs available.")
        return
    
    # Group by cluster
    clusters = set(l["cluster_id"] for l in acs_logs)
    
    for cluster_id in sorted(clusters):
        st.markdown(f"#### Cluster {cluster_id}")
        
        cluster_logs = [l for l in acs_logs if l["cluster_id"] == cluster_id]
        
        # Pheromone initialization
        init_logs = [l for l in cluster_logs if l.get("step") == "init_pheromone"]
        if init_logs:
            init = init_logs[0]
            st.markdown(f"**Pheromone Init:** τ₀ = {init['tau0']} ({init['formula']})")
        
        # Iteration details
        iterations = set(l.get("iteration") for l in cluster_logs if l.get("iteration"))
        
        for iteration in sorted(iterations):
            with st.expander(f"Iteration {iteration}", expanded=False):
                iter_logs = [l for l in cluster_logs if l.get("iteration") == iteration]
                
                # Ant route construction
                ant_logs = [l for l in iter_logs if "ant" in l and "step" in l and "probabilities" in l]
                if ant_logs:
                    st.markdown("**Route Construction:**")
                    df = pd.DataFrame([{
                        "Ant": l["ant"],
                        "Step": l["step"],
                        "From": l["from_node"],
                        "q": l["random_q"],
                        "Decision": l["decision"],
                        "Selected": l["selected"],
                        "Probabilities": str(l["probabilities"])[:50] + "..."
                    } for l in ant_logs])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Route evaluation
                route_logs = [l for l in iter_logs if "route" in l and "objective" in l]
                if route_logs:
                    st.markdown("**Route Evaluation:**")
                    df = pd.DataFrame([{
                        "Ant": l["ant"],
                        "Route": str(l["route"]),
                        "Distance": l["distance"],
                        "Service Time": l["service_time"],
                        "TW Violation": l["tw_violation"],
                        "Objective": l["objective"]
                    } for l in route_logs])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Iteration summary
                summary_logs = [l for l in iter_logs if l.get("step") == "iteration_summary"]
                if summary_logs:
                    s = summary_logs[0]
                    st.success(f"**Best Route:** {s['best_route']} | Distance: {s['best_distance']} | Objective: {s['best_objective']}")


def _display_rvnd_inter_iterations(logs: List[Dict]) -> None:
    """Display RVND inter-route iterations."""
    st.markdown("### 🔄 RVND Inter-Route Iterations")
    
    inter_logs = [l for l in logs if l.get("phase") == "RVND-INTER"]
    
    if not inter_logs:
        st.info("No RVND inter-route iterations (no improvements found or single route).")
        return
    
    df = pd.DataFrame([{
        "Iteration": l["iteration"],
        "Neighborhood": l["neighborhood"],
        "Distance Before": l["distance_before"],
        "Distance After": l["distance_after"],
        "Accepted": "✅" if l["accepted"] else "❌",
        "Best Move": str(l.get("best_move", "N/A"))
    } for l in inter_logs])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _display_rvnd_intra_iterations(logs: List[Dict]) -> None:
    """Display RVND intra-route iterations."""
    st.markdown("### 🔁 RVND Intra-Route Iterations")
    
    intra_logs = [l for l in logs if l.get("phase") == "RVND-INTRA"]
    
    if not intra_logs:
        st.info("No RVND intra-route iterations (no improvements found).")
        return
    
    # Group by cluster
    clusters = set(l["cluster_id"] for l in intra_logs)
    
    for cluster_id in sorted(clusters):
        with st.expander(f"Cluster {cluster_id}", expanded=True):
            cluster_logs = [l for l in intra_logs if l["cluster_id"] == cluster_id]
            df = pd.DataFrame([{
                "Iteration": l["iteration"],
                "Neighborhood": l["neighborhood"],
                "Sequence Before": str(l["sequence_before"]),
                "Sequence After": str(l["sequence_after"]),
                "Dist Before": l["distance_before"],
                "Dist After": l["distance_after"],
                "Accepted": "✅" if l["accepted"] else "❌"
            } for l in cluster_logs])
            st.dataframe(df, use_container_width=True, hide_index=True)


def _display_vehicle_assignment(logs: List[Dict]) -> None:
    """Display vehicle reassignment."""
    st.markdown("### 🚛 Vehicle Reassignment")
    
    vehicle_logs = [l for l in logs if l.get("phase") == "VEHICLE_REASSIGN"]
    
    if vehicle_logs:
        df = pd.DataFrame([{
            "Cluster": l["cluster_id"],
            "Demand": l["demand"],
            "Old Vehicle": l["old_vehicle"],
            "New Vehicle": l["new_vehicle"],
            "Reason": l["reason"]
        } for l in vehicle_logs])
        st.dataframe(df, use_container_width=True, hide_index=True)


def _display_final_results(result: Dict[str, Any]) -> None:
    """Display final routes and costs."""
    st.markdown("### 📊 Final Results")
    
    routes = result.get("routes", [])
    costs = result.get("costs", {})
    
    # Routes summary
    if routes:
        st.markdown("**Final Routes:**")
        df = pd.DataFrame([{
            "Cluster": r["cluster_id"],
            "Vehicle": r["vehicle_type"],
            "Route": str(r["sequence"]),
            "Distance": r["total_distance"],
            "Service Time": r["total_service_time"],
            "TW Violation": r["total_tw_violation"],
            "Demand": r["total_demand"]
        } for r in routes])
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Costs
    if costs:
        st.markdown("**Cost Breakdown:**")
        
        breakdown = costs.get("breakdown", [])
        if breakdown:
            df = pd.DataFrame([{
                "Cluster": c["cluster_id"],
                "Vehicle": c["vehicle_type"],
                "Fixed Cost": f"Rp {c['fixed_cost']:,.0f}",
                "Variable Cost": f"Rp {c['variable_cost']:,.0f}",
                "Total Cost": f"Rp {c['total_cost']:,.0f}"
            } for c in breakdown])
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Fixed Cost", f"Rp {costs.get('total_fixed_cost', 0):,.0f}")
        with col2:
            st.metric("Total Variable Cost", f"Rp {costs.get('total_variable_cost', 0):,.0f}")
        with col3:
            st.metric("TOTAL COST", f"Rp {costs.get('total_cost', 0):,.0f}")


def _display_validation(result: Dict[str, Any]) -> None:
    """Display validation against Word document."""
    st.markdown("### ✅ Validation Against Word Document")
    
    validation = result.get("validation", [])
    all_valid = result.get("all_valid", False)
    
    if all_valid:
        st.success("🎉 ALL ROUTES MATCH THE WORD DOCUMENT!")
    else:
        st.error("⚠️ SOME ROUTES DO NOT MATCH - SEE DETAILS BELOW")
    
    if validation:
        df = pd.DataFrame([{
            "Cluster": v["cluster_id"],
            "Expected Sequence": str(v["sequence_expected"]),
            "Actual Sequence": str(v["sequence_actual"]),
            "Seq Match": "✅" if v["sequence_match"] else "❌",
            "Expected Dist": v["distance_expected"],
            "Actual Dist": v["distance_actual"],
            "Dist Match": "✅" if v["distance_match"] else "❌",
            "Valid": "✅" if v["valid"] else "❌"
        } for v in validation])
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_academic_replay() -> None:
    """Main render function for Academic Replay tab."""
    st.header("📚 Academic Replay Mode")
    st.markdown("*Validasi terhadap dokumen 'Hitung Manual MFVRPTE RVND'*")
    
    st.divider()
    
    # Run button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Run Academic Replay", type="primary"):
            with st.spinner("Running academic replay..."):
                try:
                    # Import and run
                    import sys
                    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
                    from academic_replay import run_academic_replay
                    
                    result = run_academic_replay()
                    st.session_state["academic_result"] = result
                    st.success("Academic replay completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        st.info("Klik untuk menjalankan validasi akademik sesuai dokumen Word.")
    
    st.divider()
    
    # Load and display results
    result = st.session_state.get("academic_result") or _load_academic_results()
    
    if not result:
        st.warning("Belum ada hasil. Klik 'Run Academic Replay' untuk memulai.")
        return
    
    logs = result.get("iteration_logs", [])
    
    # Create tabs for each phase
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📐 Sweep", 
        "🔗 NN", 
        "🐜 ACS", 
        "🔄 RVND-Inter",
        "🔁 RVND-Intra",
        "📊 Final Results",
        "✅ Validation"
    ])
    
    with tab1:
        _display_sweep_iterations(logs)
    
    with tab2:
        _display_nn_iterations(logs)
    
    with tab3:
        _display_acs_iterations(logs)
    
    with tab4:
        _display_rvnd_inter_iterations(logs)
    
    with tab5:
        _display_rvnd_intra_iterations(logs)
    
    with tab6:
        _display_vehicle_assignment(logs)
        _display_final_results(result)
    
    with tab7:
        _display_validation(result)
