import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_part_d():
    df = pd.read_csv("data/modelling_table.csv")
    hubs_static = pd.read_csv("data/hubs.csv")
    
    # 1. Compute foundational hub operation summaries
    hub_profile = df.groupby("hub_id").agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    ).round(4).reset_index()
    
    hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)
    hub_profile.to_csv("outputs/hub_profile.csv", index=False)
    
    # 2. Distance clustering vectorization
    cluster_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(hub_profile[cluster_cols])
    
    # 3. Structural silhouette checks over pairwise clusters
    separation_records = []
    n_hubs = len(hub_profile)
    
    for k_val in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_matrix)
        
        # Calculate cross-over-within cluster distance spreads
        across_dists = []
        within_dists = []
        
        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                dist = np.linalg.norm(scaled_matrix[i] - scaled_matrix[j])
                if labels[i] == labels[j]:
                    within_dists.append(dist)
                else:
                    across_dists.append(dist)
                    
        ratio = np.mean(across_dists) / np.mean(within_dists)
        
        counts = pd.Series(labels).value_counts().tolist()
        sizes_str = "-".join(map(str, counts))
        
        separation_records.append({"K": k_val, "across_over_within": round(ratio, 4), "cluster_sizes": sizes_str})
        
    pd.DataFrame(separation_records).to_csv("outputs/part_d_separation.csv", index=False)
    
    # 4. Map operational cluster groups to target business labels (Standard, Watch, Escalate)
    km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    hub_profile["cluster_raw"] = km_final.fit_predict(scaled_matrix)
    
    cluster_delays = hub_profile.groupby("cluster_raw")["delay_rate"].mean().sort_values()
    # Define mapping strategy linked directly to performance delays
    name_mapping = {
        cluster_delays.index[0]: "Standard",
        cluster_delays.index[1]: "Watch",
        cluster_delays.index[2]: "Escalate"
    }
    hub_profile["segment"] = hub_profile["cluster_raw"].map(name_mapping)
    
    # Structural consolidation with original context maps
    segments_output = hub_profile[["hub_id", "segment"]].merge(hubs_static[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
    segments_output.to_csv("outputs/hub_segments.csv", index=False)
    
    # Profile summaries across business tracks
    segment_profile = hub_profile.groupby("segment").agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    ).round(4).reset_index()
    
    # Order tracking records consistently based on network delays
    segment_profile["sort_order"] = segment_profile["segment"].map({"Standard": 0, "Watch": 1, "Escalate": 2})
    segment_profile = segment_profile.sort_values("sort_order").drop(columns=["sort_order"]).reset_index(drop=True)
    segment_profile.to_csv("outputs/segment_profile.csv", index=False)
    
    # 5. Extract structural administrative label deviations
    tier_map = {1: "Standard", 2: "Watch", 3: "Escalate"}
    mismatches = segments_output[segments_output["segment"] != segments_output["hub_tier"].map(tier_map)]
    mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

if __name__ == "__main__":
    run_part_d()
