import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.shipment_prep import clean_shipments, FEATURES

def run_package_model():
    os.makedirs("models", exist_ok=True)
    
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # Reconstruct training split topology identically
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Fetch runtime environment selections
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    
    # Instantiates the designated production configuration mapping
    max_d = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
    rf = RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=max_d,
        min_samples_leaf=int(config["min_samples_leaf"]),
        random_state=42
    )
    
    production_pipeline = Pipeline([("selector", SelectKBest(f_classif, k=7)), ("model", rf)])
    production_pipeline.fit(X_train, y_train)
    
    # Export system states to standard formats
    joblib.dump(production_pipeline, "models/delay_model.pkl")
    pd.DataFrame({"threshold": [config["threshold"]]}).to_csv("models/delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
    
    raw_shipments = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw_shipments)
    pd.DataFrame(list(medians.items()), columns=["column", "median"]).to_csv("models/impute_values.csv", index=False)
    
    # Operational runtime integrity verification loops
    reloaded_pipeline = joblib.load("models/delay_model.pkl")
    
    prob_in_memory = production_pipeline.predict_proba(X_test)[:, 1]
    prob_reloaded = reloaded_pipeline.predict_proba(X_test)[:, 1]
    
    max_diff = np.max(np.abs(prob_in_memory - prob_reloaded))
    
    reload_check = {
        "test_rows_scored": len(X_test),
        "reload_max_abs_probability_diff": float(max_diff),
        "threshold": float(config["threshold"])
    }
    pd.DataFrame(list(reload_check.items()), columns=["metric", "value"]).to_csv("outputs/part_d_reload_check.csv", index=False)

if __name__ == "__main__":
    run_package_model()
