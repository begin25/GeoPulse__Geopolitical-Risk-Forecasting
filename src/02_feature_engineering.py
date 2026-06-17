import pandas as pd
import numpy as np
import os

def create_pair_id(row):
    pair = sorted([str(row['Actor1CountryCode']), str(row['Actor2CountryCode'])])
    return f"{pair[0]}-{pair[1]}"

def main():
    print("Loading raw data...")
    try:
        df = pd.read_csv('data/raw/conflict_events_raw.csv')
    except FileNotFoundError:
        print("Raw data not found. Run 01_data_collection.py first.")
        return

    df['SQLDATE'] = pd.to_datetime(df['SQLDATE'].astype(str), format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['SQLDATE'])

    numeric_cols = ['QuadClass', 'GoldsteinScale', 'NumMentions', 'AvgTone']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    df = df[df['Actor1CountryCode'] != df['Actor2CountryCode']]
    df['Pair_ID'] = df.apply(create_pair_id, axis=1)

    print("Aggregating daily stats per country pair...")
    df['IsConflict'] = (df['QuadClass'] >= 3).astype(int)

    daily = df.groupby(['SQLDATE', 'Pair_ID']).agg(
        Total_Events=('SQLDATE', 'count'),
        Conflict_Events=('IsConflict', 'sum'),
        Avg_Goldstein=('GoldsteinScale', 'mean'),
        Avg_Tone=('AvgTone', 'mean'),
        Total_Mentions=('NumMentions', 'sum')
    ).reset_index()

    # Unstack and restack to guarantee every pair has a continuous daily timeline (fills missing days with 0)
    daily = daily.set_index(['SQLDATE', 'Pair_ID']).unstack(fill_value=0).asfreq('D', fill_value=0).stack().reset_index()
    daily = daily.sort_values(by=['Pair_ID', 'SQLDATE']).reset_index(drop=True)

    print("Engineering dynamic predictive features...")
    
    daily['Events_3d_sum'] = daily.groupby('Pair_ID')['Total_Events'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    daily['Events_7d_sum'] = daily.groupby('Pair_ID')['Total_Events'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    daily['Events_14d_sum'] = daily.groupby('Pair_ID')['Total_Events'].transform(lambda x: x.rolling(14, min_periods=1).sum())
    
    daily['Tone_7d_mean'] = daily.groupby('Pair_ID')['Avg_Tone'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    daily['Goldstein_7d_mean'] = daily.groupby('Pair_ID')['Avg_Goldstein'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    daily['Goldstein_momentum'] = daily.groupby('Pair_ID')['Avg_Goldstein'].transform(lambda x: x.diff().fillna(0))
    daily['Tone_7d_std'] = daily.groupby('Pair_ID')['Avg_Tone'].transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))

    # Calculate 30-day Volume Baseline (V_b) - Kept as a feature so the model learns absolute scale!
    daily['V_b'] = daily.groupby('Pair_ID')['Conflict_Events'].transform(lambda x: x.rolling(30, min_periods=1).mean() * 7)

    # Reverse -> Roll -> Reverse -> Shift to look strictly into the future
    daily['V_f'] = daily.groupby('Pair_ID')['Conflict_Events'].transform(lambda x: x.iloc[::-1].rolling(7, min_periods=1).sum().iloc[::-1].shift(-1).fillna(0))
    daily['T_f'] = daily.groupby('Pair_ID')['Avg_Tone'].transform(lambda x: x.iloc[::-1].rolling(7, min_periods=1).mean().iloc[::-1].shift(-1).fillna(0))

    print("Calculating Dual-Engine Escalation Target...")
    # Extract Global Structural Parameters dynamically from the dataset
    alpha_global = daily['V_b'].median()
    v_critical = daily['V_b'].quantile(0.75)
    
    print(f"-> Global Noise Dampener (alpha): {alpha_global:.1f} events/week")
    print(f"-> Critical Volume Floor (V_critical): {v_critical:.1f} events/week")

    # ENGINE 1: Dampened Relative Breakout (Growth > 40%)
    gamma = 0.40
    daily['E_breakout'] = (((daily['V_f'] - daily['V_b']) / (daily['V_b'] + alpha_global)) > gamma).astype(int)

    # ENGINE 2: Absolute Hostility Floor (Massive Volume + Plunging Tone)
    t_critical = -4.0
    daily['E_hostility'] = ((daily['V_f'] >= v_critical) & (daily['T_f'] <= t_critical)).astype(int)

    # UNIFIED TARGET (OR GATE)
    daily['Target_Escalation'] = (daily['E_breakout'] | daily['E_hostility']).astype(int)

    # Strict Data Leakage Masking: Wipe the exact final 7 days of the whole dataset
    max_global_date = daily['SQLDATE'].max()
    cutoff_date = max_global_date - pd.Timedelta(days=7)
    daily.loc[daily['SQLDATE'] >= cutoff_date, 'Target_Escalation'] = np.nan

    # Drop target calculation components to prevent data leakage into features
    drop_cols = ['V_f', 'T_f', 'E_breakout', 'E_hostility']
    final_features = daily.drop(columns=drop_cols).sort_values('SQLDATE').reset_index(drop=True)

    os.makedirs('data/processed', exist_ok=True)
    out_path = 'data/processed/features_labels.csv'
    final_features.to_csv(out_path, index=False)
    
    valid_targets = final_features['Target_Escalation'].dropna()
    print(f"Feature engineering complete! Shape: {final_features.shape}")
    print(f"Global Escalation Base Rate: {valid_targets.mean() * 100:.1f}%")

if __name__ == "__main__":
    main()
"""v1, rip"""

# import pandas as pd
# import numpy as np
# import os

# def create_pair_id(row):
#     pair = sorted([str(row['Actor1CountryCode']), str(row['Actor2CountryCode'])])
#     return f"{pair[0]}-{pair[1]}"

# def main():
#     print("Loading raw data...")
#     try:
#         df = pd.read_csv('data/raw/conflict_events_raw.csv')
#     except FileNotFoundError:
#         print("Raw data not found. Run 01_data_collection.py first.")
#         return

#     # 1. Cleaning & Type Casting
#     df['SQLDATE'] = pd.to_datetime(df['SQLDATE'].astype(str), format='%Y%m%d', errors='coerce')
#     df = df.dropna(subset=['SQLDATE'])

#     numeric_cols = ['QuadClass', 'GoldsteinScale', 'NumMentions', 'AvgTone']
#     df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

#     # 2. Filter valid cross-border events
#     df = df[df['Actor1CountryCode'] != df['Actor2CountryCode']]
#     df['Pair_ID'] = df.apply(create_pair_id, axis=1)

#     print("Aggregating daily stats per country pair...")
#     df['IsConflict'] = (df['QuadClass'] >= 3).astype(int)

#     daily = df.groupby(['SQLDATE', 'Pair_ID']).agg(
#         Total_Events=('SQLDATE', 'count'),
#         Conflict_Events=('IsConflict', 'sum'),
#         Avg_Goldstein=('GoldsteinScale', 'mean'),
#         Avg_Tone=('AvgTone', 'mean'),
#         Total_Mentions=('NumMentions', 'sum')
#     ).reset_index()

#     daily = daily.sort_values(by=['Pair_ID', 'SQLDATE'])
#     max_global_date = daily['SQLDATE'].max()

#     features_df_list = []

#     print("Engineering features and calculating Matrix Target...")
#     for pair_id, group in daily.groupby('Pair_ID'):
#         group = group.set_index('SQLDATE')
#         group = group.asfreq('D')
#         group['Pair_ID'] = pair_id
#         group = group.fillna(0)

#         # Compute Historical Rolling Features
#         for w in [3, 7, 14]:
#             group[f'Events_{w}d_sum'] = group['Total_Events'].rolling(window=w, min_periods=1).sum()
#             group[f'Goldstein_{w}d_mean'] = group['Avg_Goldstein'].rolling(window=w, min_periods=1).mean()
#             group[f'Tone_{w}d_mean'] = group['Avg_Tone'].rolling(window=w, min_periods=1).mean()

#         group['Goldstein_momentum'] = group['Avg_Goldstein'].diff().fillna(0)
#         group['Tone_7d_std'] = group['Avg_Tone'].rolling(window=7, min_periods=1).std().fillna(0)

#         # --- THE NEW VELOCITY & SENTIMENT TARGET LOGIC ---
        
#         # Look-ahead features (Reverse -> Roll -> Reverse -> Shift)
#         future_sum = group['Conflict_Events'].iloc[::-1].rolling(window=7, min_periods=1).sum().iloc[::-1].shift(-1)
#         future_tone = group['Avg_Tone'].iloc[::-1].rolling(window=7, min_periods=1).mean().iloc[::-1].shift(-1)
        
#         group['Future_7d_Conflict'] = future_sum.fillna(0)
#         group['Future_7d_Tone'] = future_tone.fillna(0)

#         # Baselines
#         group['Baseline_30d_mean'] = group['Conflict_Events'].rolling(window=30, min_periods=1).mean().fillna(0)
#         group['Baseline_30d_Tone_mean'] = group['Avg_Tone'].rolling(window=30, min_periods=1).mean().fillna(0)
        
#         # Pillar 1: Dampened Volumetric Acceleration
#         V_b = group['Baseline_30d_mean'] * 7
#         V_f = group['Future_7d_Conflict']
#         gamma = 0.40  # 40% growth required
#         alpha = 35    # Noise dampener
#         V_acc = ((V_f - V_b) / (V_b + alpha)) > gamma
        
#         # Pillar 2: Sentiment Crash (For saturated corridors)
#         delta = 1.5
#         T_critical = -5.0
#         T_f = group['Future_7d_Tone']
#         mu_T30 = group['Baseline_30d_Tone_mean']
        
#         S_crash = ((T_f - mu_T30) < -delta) & (T_f < T_critical) & (V_f >= 25)

#         # OR Gate: Either a structural volume shock or a hostile sentiment crash
#         group['Target_Escalation'] = (V_acc | S_crash).astype(int)

#         features_df_list.append(group.reset_index())

#     final_features = pd.concat(features_df_list, ignore_index=True)

#     # Prevent Data Leakage: Mask target for the final 7 days where the future hasn't occurred
#     cutoff_date = max_global_date - pd.Timedelta(days=7)
#     final_features.loc[final_features['SQLDATE'] >= cutoff_date, 'Target_Escalation'] = np.nan

#     final_features = final_features.sort_values('SQLDATE').reset_index(drop=True)

#     out_path = 'data/processed/features_labels.csv'
#     final_features.to_csv(out_path, index=False)
#     print(f"Feature engineering complete! Shape: {final_features.shape}")

# if __name__ == "__main__":
#     main()