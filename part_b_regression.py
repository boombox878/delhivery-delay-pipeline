import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES

def run_part_b():
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["delivery_hours"]
    
    # 1. Leakage-Free Stratum Splitting
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 2. Transform Scaling verification metrics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    checks = {
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "train_scaled_max_abs_mean": np.max(np.abs(np.mean(X_train_scaled, axis=0))),
        "test_scaled_max_abs_mean": np.max(np.abs(np.mean(X_test_scaled, axis=0)))
    }
    pd.DataFrame(list(checks.items()), columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)
    
    # 3. Decision Tree Tuning Mechanics
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_records = []
    best_depth = None
    min_val_rmse = float("inf")
    
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        p_train = dt.predict(X_train)
        p_val = dt.predict(X_val)
        
        r_train = np.sqrt(mean_squared_error(y_train, p_train))
        r_val = np.sqrt(mean_squared_error(y_val, p_val))
        
        depth_label = "unlimited" if d is None else d
        tree_records.append({"max_depth": depth_label, "train_rmse": round(r_train, 4), "val_rmse": round(r_val, 4)})
        
        if r_val < min_val_rmse:
            min_val_rmse = r_val
            best_depth = d
            
    pd.DataFrame(tree_records).to_csv("outputs/part_b_tree_depth.csv", index=False)
    
    # 4. Multi-algorithm benchmarking configuration
    models = {
        "mean_baseline": None,
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    metrics_records = []
    val_rmses = {}
    
    for name, model in models.items():
        if name == "mean_baseline":
            preds = np.full(shape=len(y_val), fill_value=y_train.mean())
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)
        
        metrics_records.append({
            "model": name, "split": "val", "MAE": round(mae, 4), 
            "MSE": round(mse, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)
        })
        if name != "mean_baseline":
            val_rmses[name] = rmse
            
    # 5. Learning rate tuning spectrum for boosting models
    lr_records = []
    for lr in [0.01, 0.1, 1.0]:
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gbr.fit(X_train, y_train)
        preds = gbr.predict(X_val)
        lr_records.append({
            "learning_rate": lr,
            "val_rmse": round(np.sqrt(mean_squared_error(y_val, preds)), 4),
            "val_r2": round(r2_score(y_val, preds), 4)
        })
    pd.DataFrame(lr_records).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    
    # 6. CV comparisons and Test evaluation rules
    sorted_candidates = sorted(val_rmses.items(), key=lambda item: item[1])
    top_two = [sorted_candidates[0][0], sorted_candidates[1][0]]
    
    X_cv_comb = pd.concat([X_train, X_val])
    y_cv_comb = pd.concat([y_train, y_val])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_records = []
    for name in top_two:
        m_obj = models[name]
        scores = cross_val_score(m_obj, X_cv_comb, y_cv_comb, cv=kf, scoring="r2")
        cv_records.append({"model": name, "mean_r2": round(np.mean(scores), 4), "std_r2": round(np.std(scores), 4)})
    pd.DataFrame(cv_records).to_csv("outputs/part_b_cv.csv", index=False)
    
    # Select winning model variant based strictly on validation error
    final_model_name = sorted_candidates[0][0]
    final_model = models[final_model_name]
    
    # Holdout validation testing step
    t_preds = final_model.predict(X_test)
    t_mae = mean_absolute_error(y_test, t_preds)
    t_mse = mean_squared_error(y_test, t_preds)
    t_rmse = np.sqrt(t_mse)
    t_r2 = r2_score(y_test, t_preds)
    
    metrics_records.append({
        "model": final_model_name, "split": "test", "MAE": round(t_mae, 4),
        "MSE": round(t_mse, 4), "RMSE": round(t_rmse, 4), "R2": round(t_r2, 4)
    })
    pd.DataFrame(metrics_records).to_csv("outputs/part_b_metrics.csv", index=False)

if __name__ == "__main__":
    run_part_b()
