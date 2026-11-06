import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, NOISE_COLUMNS

def calculate_metrics(y_true, y_pred, model_name, split_name="val"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": model_name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4)
    }

def run_part_c():
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # 1. Stratified Multi-Split partitions
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 2. Baseline & structural benchmark fits
    models_out = []
    
    # Blind baseline rule
    blind_preds = np.zeros(len(y_val))
    models_out.append(calculate_metrics(y_val, blind_preds, "blind_all_on_time"))
    
    # Linear Classification setup
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    lr_pipe.fit(X_train, y_train)
    models_out.append(calculate_metrics(y_val, lr_pipe.predict(X_val), "logistic_regression"))
    
    # Evaluates structural variations across depth trees
    best_dt_d, best_dt_f1 = None, -1
    dt_candidates = {}
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        m = calculate_metrics(y_val, dt.predict(X_val), f"decision_tree_depth_{d}")
        dt_candidates[d] = m
        if m["f1"] > best_dt_f1:
            best_dt_f1 = m["f1"]
            best_dt_d = d
    models_out.append(dt_candidates[best_dt_d])
    
    # Reference baseline random forest
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    models_out.append(calculate_metrics(y_val, rf_def.predict(X_val), "random_forest_default"))
    pd.DataFrame(models_out).to_csv("outputs/part_c_models.csv", index=False)
    
    # 3. Variance & F-Score Signal Isolations
    selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
    f_scores = selector_all.scores_
    f_ranks = pd.Series(f_scores).rank(ascending=False, method="min").astype(int).values
    
    feature_scores = pd.DataFrame({
        "feature": FEATURES, "f_score": np.round(f_scores, 4), "f_rank": f_ranks,
        "rf_importance": np.round(rf_def.feature_importances_, 4),
        "is_noise": [f in NOISE_COLUMNS for f in FEATURES]
    })
    feature_scores.to_csv("outputs/part_c_feature_scores.csv", index=False)
    
    # Dimension reduction benchmarks
    k_records = []
    for k_val in [5, 7, 10, 12, 17]:
        k_pipe = Pipeline([("selector", SelectKBest(f_classif, k=k_val)), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
        k_pipe.fit(X_train, y_train)
        k_records.append({"k": k_val, "val_f1": round(f1_score(y_val, k_pipe.predict(X_val), zero_division=0), 4)})
    pd.DataFrame(k_records).to_csv("outputs/part_c_selectk.csv", index=False)
    
    # 4. Hyperparameter Tuning execution maps
    # Use fixed feature selector scaling from here out (k=7)
    sel = SelectKBest(f_classif, k=7)
    X_tr_sel = sel.fit_transform(X_train, y_train)
    X_val_sel = sel.transform(X_val)
    
    param_grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
    
    gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5).fit(X_tr_sel, y_train)
    rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42).fit(X_tr_sel, y_train)
    
    # Sub-baseline tracking for feature controls
    rf_same_7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr_sel, y_train)
    
    tuning_records = [
        {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135, "best_params": str(gs.best_params_), "best_cv_accuracy": round(gs.best_score_, 4), "val_f1": round(f1_score(y_val, gs.predict(X_val_sel)), 4)},
        {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45, "best_params": str(rs.best_params_), "best_cv_accuracy": round(rs.best_score_, 4), "val_f1": round(f1_score(y_val, rs.predict(X_val_sel)), 4)},
        {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1, "best_params": "defaults", "best_cv_accuracy": "", "val_f1": round(f1_score(y_val, rf_same_7.predict(X_val_sel)), 4)}
    ]
    pd.DataFrame(tuning_records).to_csv("outputs/part_c_tuning.csv", index=False)
    
    # 5. Extract structural configurations
    tuning_df = pd.DataFrame(tuning_records)
    best_idx = tuning_df["val_f1"].idxmax()
    winner_name = tuning_df.loc[best_idx, "search"]
    
    if winner_name == "grid_search":
        final_model_obj = gs.best_estimator_
        p_dict = gs.best_params_
    elif winner_name == "randomized_search":
        final_model_obj = rs.best_estimator_
        p_dict = rs.best_params_
    else:
        final_model_obj = rf_same_7
        p_dict = {"n_estimators": 100, "max_depth": "unlimited", "min_samples_leaf": 1}
        
    final_pipeline = Pipeline([("selector", SelectKBest(f_classif, k=7)), ("model", final_model_obj)])
    final_pipeline.fit(X_train, y_train)
    
    # 6. Operations Risk Probability Boundary Searches
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_records = []
    chosen_threshold = 0.1
    
    for t in thresholds:
        t_preds = (val_probs >= t).astype(int)
        m = calculate_metrics(y_val, t_preds, f"thresh_{t}")
        thresh_records.append({
            "threshold": t, "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"],
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]
        })
        # Capture operations risk parameters (Recall >= 0.80)
        if m["recall"] >= 0.80:
            chosen_threshold = t
            
    pd.DataFrame(thresh_records).to_csv("outputs/part_c_thresholds.csv", index=False)
    
    # 7. Final Holdout Evaluations
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)
    final_test_metrics = calculate_metrics(y_test, test_preds, "final_model_test")
    
    pd.DataFrame([{
        "threshold": chosen_threshold, "TN": final_test_metrics["TN"], "FP": final_test_metrics["FP"],
        "FN": final_test_metrics["FN"], "TP": final_test_metrics["TP"], "accuracy": final_test_metrics["accuracy"],
        "precision": final_test_metrics["precision"], "recall": final_test_metrics["recall"], "f1": final_test_metrics["f1"]
    }]).to_csv("outputs/part_c_final_test.csv", index=False)
    
    pd.DataFrame([{
        "selector_k": 7, "n_estimators": p_dict.get("n_estimators", 100),
        "max_depth": p_dict.get("max_depth", "unlimited"), "min_samples_leaf": p_dict.get("min_samples_leaf", 1),
        "threshold": chosen_threshold, "chosen_from": winner_name
    }]).to_csv("outputs/part_c_final_config.csv", index=False)

if __name__ == "__main__":
    run_part_c()
