import sys
import data_analysis
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib


def calculate_historical_mean(df):
    # 1. Sort strictly by Category and Sequential Timeline
    df = df.sort_values(by=['district', 'market', 'commodity_category', 'year', 'months']).reset_index(drop=True)

    # 2. Group by location and commodity keys
    grouped = df.groupby(['district', 'market', 'commodity_category'])['arrivals_mt']

    # 3. FIXED SHIFTS: Shift by 12 months for a 1-Year Lag, 24 months for a 2-Year Lag
    df['arrival_lag_1yr'] = grouped.shift(12)  # Looks back exactly 1 year ago same month
    df['arrival_lag_2yr'] = grouped.shift(24)  # Looks back exactly 2 years ago same month

    # 4. FIXED ROLLING: Calculate true multi-year benchmarks for that specific month
    # We calculate a rolling average across a 36-month window, sampling every 12th month
    df['arrival_rolling_mean_3yr'] = (
        grouped.apply(lambda x: x.shift(12).rolling(window=36, min_periods=12).mean())
        .reset_index(level=[0, 1, 2], drop=True)
    )

    df['arrival_rolling_mean_5yr'] = (
        grouped.apply(lambda x: x.shift(12).rolling(window=60, min_periods=12).mean())
        .reset_index(level=[0, 1, 2], drop=True)
    )

    df_final = df

    return df_final


def create_model(df):
    cat_features_list = ['district', 'market', 'commodity_category', 'months']

    train_mask = df['year'] <= 2025
    test_period = df['year'] == 2026

    X_train = df.loc[train_mask, cat_features_list]
    y_train = df.loc[train_mask, 'arrivals_mt']

    X_test = df.loc[test_period, cat_features_list]
    y_test = df.loc[test_period, 'arrivals_mt']

    # 2. CREATE SAMPLE WEIGHTS
    # Give higher weight to larger transactions so the model doesn't ignore them
    # This forces the trees to focus heavily on reducing error for big arrival days
    sample_weights = np.where(y_train > 100, 10.0, 1.0)
    sample_weights = np.where(y_train > 1000, 50.0, sample_weights)
    sample_weights = np.where(y_train > 10000, 250.0, sample_weights)

    # 3. Train on the RAW scale using RMSE (No log transforms!)
    cat = CatBoostRegressor(
        iterations=2424,
        learning_rate=0.05,
        depth=10,
        loss_function='RMSE',
        random_seed=42
    )

    # Pass the sample weights here
    cat.fit(X_train, y_train, cat_features_list, sample_weight=sample_weights, verbose=200)

    # 4. Predict directly on the true scale
    final_predictions = cat.predict(X_test)
    final_predictions = np.clip(final_predictions, 0, None)  # No negative arrivals
    final_actuals = y_test.values

    # 5. Evaluate
    real_world_rmse = np.sqrt(mean_squared_error(final_actuals, final_predictions))
    real_world_mae = mean_absolute_error(final_actuals, final_predictions)

    print("--- WEIGHTED RAW SCALE METRICS ---")
    print(f"Actual Max 2026 Arrival: {final_actuals.max():.2f}")
    print(f"Predicted Max 2026 Arrival: {final_predictions.max():.2f} MT")
    print(f"True Real-World RMSE: {real_world_rmse:.2f}")
    print(f"True Real-World MAE: {real_world_mae:.2f} MT")

    return cat


def main():
    # import df
    df = data_analysis.main()

    # remove commodity as we have created commodity category
    df.drop(['commodity'], axis=1, inplace=True)

    # convert date to months
    df['months'] = df['date'].dt.month
    df.drop(['date'], axis=1, inplace=True)

    df = calculate_historical_mean(df)

    df['commodity_category'] = df['commodity_category'].astype('category')

    # create model
    cat_model = create_model(df)

    # save model
    joblib.dump(cat_model, 'agri_cat_model.joblib')

    return df


if __name__ == "__main__":
    sys.exit(main())

