import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

def run_part_a():
    os.makedirs("outputs", exist_ok=True)
    
    # Load raw assets
    raw_shipments = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    
    # Pre-cleaning instrumentation values
    raw_rows = len(raw_shipments)
    dedup_temp = raw_shipments.drop_duplicates()
    duplicate_rows_removed = raw_rows - len(dedup_temp)
    
    missing_dist = dedup_temp["distance_km"].isna().sum()
    missing_val = dedup_temp["declared_value"].isna().sum()
    missing_pay = dedup_temp["payment_mode"].isna().sum()
    
    # Exclude sentinels (>100) from weight null count
    missing_wgt = dedup_temp["parcel_weight_kg"].isna().sum() 
    sentinels = (dedup_temp["parcel_weight_kg"] > 100).sum()
    
    spellings_before = dedup_temp["service_type"].astype(str).nunique()
    
    # Execute core transformation pipeline
    clean_df, medians = clean_shipments(raw_shipments)
    spellings_after = clean_df["service_type"].nunique()
    
    modelling_table = build_features(clean_df, hubs)
    
    # Target labeling logic
    modelling_table["is_delayed"] = (modelling_table["delivery_hours"] > modelling_table["promised_hours"]).astype(int)
    modelling_table.to_csv("data/modelling_table.csv", index=False)
    
    # Outlier threshold definitions using 1.5 * IQR rules
    q1 = modelling_table["delivery_hours"].quantile(0.25)
    q3 = modelling_table["delivery_hours"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = ((modelling_table["delivery_hours"] < lower_bound) | (modelling_table["delivery_hours"] > upper_bound)).sum()
    
    # 1. Output metrics check table
    checks = {
        "raw_rows": raw_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "rows_after_dedup": len(dedup_temp),
        "missing_distance_km_before": missing_dist,
        "missing_parcel_weight_before": missing_wgt,
        "sentinel_weight_rows": sentinels,
        "missing_declared_value_before": missing_val,
        "missing_payment_mode_before": missing_pay,
        "service_type_spellings_before": spellings_before,
        "service_type_spellings_after": spellings_after,
        "median_distance_km": medians["distance_km"],
        "median_parcel_weight_kg": medians["parcel_weight_kg"],
        "median_declared_value_inr": medians["declared_value_inr"],
        "rows_after_hub_merge": len(modelling_table),
        "nulls_in_features_and_target": modelling_table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum(),
        "delayed_count": modelling_table["is_delayed"].sum(),
        "delay_rate": round(modelling_table["is_delayed"].mean(), 4),
        "hours_q1": round(q1, 4),
        "hours_q3": round(q3, 4),
        "hours_lower_bound": round(lower_bound, 4),
        "hours_upper_bound": round(upper_bound, 4),
        "hours_outlier_count": outlier_count,
        "declared_value_mean_inr": round(modelling_table["declared_value_inr"].mean(), 4),
        "declared_value_median_inr": round(modelling_table["declared_value_inr"].median(), 4),
        "log_declared_value_mean": round(modelling_table["log_declared_value"].mean(), 4),
        "log_declared_value_median": round(modelling_table["log_declared_value"].median(), 4)
    }
    
    pd.DataFrame(list(checks.items()), columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)
    
    # 2. Distance Bins segmentation & summary profiles
    bins = [0, 200, 400, 600, 2000]
    labels = ["0-200", "201-400", "401-600", "601+"]
    modelling_table["distance_band"] = pd.cut(modelling_table["distance_km"], bins=bins, labels=labels, include_lowest=True)
    
    summary_rows = []
    for dim, col in [("service_type", "service_type"), ("hub_tier", "hub_tier"), ("distance_band", "distance_band")]:
        grouped = modelling_table.groupby(col, observed=False)
        for grp, data in grouped:
            summary_rows.append({
                "dimension": dim,
                "group": grp,
                "shipments": len(data),
                "delayed": data["is_delayed"].sum(),
                "delay_rate": round(data["is_delayed"].mean(), 4),
                "mean_hours": round(data["delivery_hours"].mean(), 4)
            })
    pd.DataFrame(summary_rows).to_csv("outputs/part_a_delay_by_group.csv", index=False)
    
    # 3. Numeric correlation matrices
    numeric_cols = modelling_table.select_dtypes(include=[np.number]).columns.tolist()
    # Ensure structural identity for tracking index
    corr_matrix = modelling_table[numeric_cols].corr().round(4)
    corr_matrix.to_csv("outputs/part_a_correlations.csv", index_label="feature")

if __name__ == "__main__":
    run_part_a()

