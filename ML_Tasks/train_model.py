import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("dataset.csv")

X = df.drop(columns=["GT_yaw", "GT_pitch", "GT_roll"])
y = df[["GT_yaw", "GT_pitch", "GT_roll"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Ridge": (
        MultiOutputRegressor(Ridge()),
        {"estimator__alpha": [0.01, 0.1, 1, 10]}
    ),

    "RandomForest": (
        MultiOutputRegressor(RandomForestRegressor(n_jobs=-1)),
        {
            "estimator__n_estimators": [150, 200],
            "estimator__max_depth": [None, 10],
        }
    ),

    "GradientBoosting": (
        MultiOutputRegressor(GradientBoostingRegressor()),
        {
            "estimator__n_estimators": [200, 300],
            "estimator__learning_rate": [0.05, 0.1],
            "estimator__max_depth": [3, 4],
        }
    ),

    "MLP": (
        MultiOutputRegressor(
            Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(max_iter=2000))
            ])
        ),
        {
            "estimator__mlp__hidden_layer_sizes": [(128,), (128,64)],
            "estimator__mlp__alpha": [0.001, 0.01],
        }
    )
}

best_model = None
best_score = float("inf")
best_name = ""

for name, (model, params) in models.items():
    print(f"\nTraining {name}...")

    search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=8,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=2,
        random_state=42
    )

    search.fit(X_train, y_train)

    y_pred = search.best_estimator_.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{name} Results:")
    print("Best Params:", search.best_params_)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    if mae < best_score:
        best_score = mae
        best_model = search.best_estimator_
        best_name = name

print(f"\nBest Model: {best_name}")
joblib.dump(best_model, "best_headpose_model.pkl")
print("Model saved as best_headpose_model.pkl")