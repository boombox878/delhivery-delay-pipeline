import pandas as pd
import joblib
from src.shipment_prep import clean_shipments, build_features

def run_predictions():
    # Load batch payloads and deployment states
    new_shipments = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    hub_segments = pd.read_csv("outputs/hub_segments.csv")
    
    pipeline = joblib.load("models/delay_model.pkl")
    threshold = pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0]
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    impute_df = pd.read_csv("models/impute_values.csv")
    
    # Structure mapping dict for standard input transformations
    medians = dict(zip(impute_df["column"], impute_df["median"]))
    
    # Standardize input vector architecture using identical training blocks
    clean_df, _ = clean_shipments(new_shipments, medians=medians)
    featurized = build_features(clean_df, hubs)
    
    # Re-verify alignment sequence against input vector mapping targets
    X_live = featurized[feature_order]
    
    # Compute active risk boundaries
    probabilities = pipeline.predict_proba(X_live)[:, 1]
    
    # Generate business payload objects
    output = pd.DataFrame({
        "shipment_id": featurized["shipment_id"],
        "hub_id": featurized["hub_id"]
    })
    
    output = output.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")
    output["delay_probability"] = pd.Series(probabilities).round(4)
    output["flagged"] = (output["delay_probability"] >= threshold).astype(int)
    
    # Route tactical operations playbooks via business criteria rules
    def assign_action(row):
        if row["flagged"] == 1:
            return "Escalate to hub manager" if row["segment"] == "Escalate" else "Proactive customer call"
        return "No action"
        
    output["action"] = output.apply(assign_action, axis=1)
    
    # Enforce exact match to original batch ingestion sequences
    final_output = new_shipments[["shipment_id"]].merge(output, on="shipment_id", how="left")
    final_output.to_csv("outputs/new_shipment_actions.csv", index=False)

if __name__ == "__main__":
    run_predictions()
