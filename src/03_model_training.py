import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

def main():
    print("Loading engineered features...")
    try:
        df = pd.read_csv('data/processed/features_labels.csv')
    except FileNotFoundError:
        print("Data not found. Run scripts 01 and 02 first.")
        return

    df['SQLDATE'] = pd.to_datetime(df['SQLDATE'])
    df = df.sort_values('SQLDATE').reset_index(drop=True)

    # Drop un-happened future targets
    df = df.dropna(subset=['Target_Escalation'])

    print("Generating generalized Pair Identity metrics...")
    pair_stats = df.groupby('Pair_ID').agg(
        Pair_Historical_Freq=('Pair_ID', 'count'),
        Pair_Escalation_Rate=('Target_Escalation', lambda x: x.mean(skipna=True))
    ).to_dict('index')

    def map_pair_stats(pid):
        stats = pair_stats.get(pid, {'Pair_Historical_Freq': 0, 'Pair_Escalation_Rate': 0.0})
        return pd.Series([stats['Pair_Historical_Freq'], stats['Pair_Escalation_Rate']])

    df[['Pair_Historical_Freq', 'Pair_Escalation_Rate']] = df['Pair_ID'].apply(map_pair_stats)

    # Isolate Predictive Features from Identifiers/Targets
    drop_cols = ['SQLDATE', 'Pair_ID', 'Target_Escalation']
    features = [c for c in df.columns if c not in drop_cols]

    X = df[features]
    y = df['Target_Escalation']

    print(f"Dataset Size: {X.shape}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)*100}")

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        eval_metric='logloss',
        random_state=42
        # Note: No scale_pos_weight used to ensure true probabilities
    )

    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, probs)
            aucs.append(auc)
            print(f"Fold {fold} - ROC AUC: {auc:.4f}")
        except ValueError:
            print(f"Fold {fold} - ROC AUC: N/A (Only one class)")

    if aucs:
        print(f"\nAverage ROC AUC: {np.mean(aucs):.4f}")

    print("\nTraining final model on full dataset...")
    model.fit(X, y)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_xgb_model.pkl')
    joblib.dump(features, 'models/feature_names.pkl')
    joblib.dump(pair_stats, 'models/pair_stats.pkl')
    print("Models and contexts saved successfully.")

if __name__ == "__main__":
    main()# import pandas as pd
# import numpy as np
# import joblib
# import os
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import classification_report, roc_auc_score
# from xgboost import XGBClassifier

# def main():
#     print("Loading engineered features...")
#     try:
#         df = pd.read_csv('data/processed/features_labels.csv')
#     except FileNotFoundError:
#         print("Data not found. Please run scripts 01 and 02 first.")
#         return

#     df['SQLDATE'] = pd.to_datetime(df['SQLDATE'])
#     df = df.sort_values('SQLDATE').reset_index(drop=True)

#     # Drop the un-happened future targets
#     df = df.dropna(subset=['Target_Escalation'])

#     print("Generating generalized Pair Identity metrics...")
#     pair_stats = df.groupby('Pair_ID').agg(
#         Pair_Historical_Freq=('Pair_ID', 'count'),
#         Pair_Escalation_Rate=('Target_Escalation', lambda x: x.mean(skipna=True))
#     ).to_dict('index')

#     def map_pair_stats(pid):
#         stats = pair_stats.get(pid, {'Pair_Historical_Freq': 0, 'Pair_Escalation_Rate': 0.0})
#         return pd.Series([stats['Pair_Historical_Freq'], stats['Pair_Escalation_Rate']])

#     df[['Pair_Historical_Freq', 'Pair_Escalation_Rate']] = df['Pair_ID'].apply(map_pair_stats)

#     # Drop Future Target Leakage columns from the feature set!
#     drop_cols = [
#         'SQLDATE', 'Pair_ID',
#         'Future_7d_Conflict', 'Future_7d_Tone', # Never let the model see the future
#         'Target_Escalation',
#         # We also drop baselines so it learns from momentum, not static absolute values
#         'Baseline_30d_mean', 'Baseline_30d_Tone_mean'
#     ]
#     features = [c for c in df.columns if c not in drop_cols]

#     X = df[features]
#     y = df['Target_Escalation']

#     print(f"Dataset Size: {X.shape}")
#     print(f"Class distribution:\n{y.value_counts(normalize=True)*100}")

#     # Probability Calibrated XGBoost
#     model = XGBClassifier(
#         n_estimators=100,
#         learning_rate=0.1,
#         max_depth=4,
#         eval_metric='logloss',
#         random_state=42
#     )

#     tscv = TimeSeriesSplit(n_splits=5)
#     fold = 1
#     aucs = []

#     for train_index, test_index in tscv.split(X):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#         model.fit(X_train, y_train)
#         probs = model.predict_proba(X_test)[:, 1]

#         try:
#             auc = roc_auc_score(y_test, probs)
#             aucs.append(auc)
#             print(f"Fold {fold} - ROC AUC: {auc:.4f}")
#         except ValueError:
#             print(f"Fold {fold} - ROC AUC: N/A (Only one class present in test fold)")

#         fold += 1

#     if aucs:
#         print(f"\nAverage ROC AUC: {np.mean(aucs):.4f}")

#     print("\nTraining final model on full dataset...")
#     model.fit(X, y)

#     os.makedirs('models', exist_ok=True)
#     joblib.dump(model, 'models/best_xgb_model.pkl')
#     joblib.dump(features, 'models/feature_names.pkl')
#     joblib.dump(pair_stats, 'models/pair_stats.pkl')
#     print("Model and context stats saved successfully to /models.")

# if __name__ == "__main__":
#     main()
