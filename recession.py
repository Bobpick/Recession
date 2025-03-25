import pandas as pd
from fredapi import Fred
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Initialize FRED with your API key
fred = Fred(api_key='YourKeyHere')  # Replace with your key

# Define series IDs
series_ids = {
    'GDP': 'GDP',
    'Yield_Spread': 'T10Y2Y',
    'Unemployment': 'UNRATE',
    'CPI': 'CPIAUCSL',
    'Recession': 'USREC',
    'Stock_Market': 'NASDAQCOM'
}

# Fetch data
data = {}
for name, series_id in series_ids.items():
    data[name] = fred.get_series(series_id)

# Combine into a DataFrame
df = pd.DataFrame(data)

# Resample to month-end
df = df.resample('ME').last()

# Interpolate GDP from quarterly to monthly
df['GDP'] = df['GDP'].interpolate(method='linear')

# Calculate GDP growth rate and NASDAQ return
df['GDP_Growth'] = df['GDP'].pct_change() * 100
df['GDP_Growth'] = df['GDP_Growth'].ffill()
df['NASDAQ_Return'] = df['Stock_Market'].pct_change() * 100
df['NASDAQ_Return'] = df['NASDAQ_Return'].ffill()

# Trim to where all core series have data
df = df.dropna(subset=['GDP_Growth', 'Yield_Spread', 'Unemployment', 'CPI', 'NASDAQ_Return', 'Recession'])

# Add recession targets for different horizons
df['Recession_Next_3M'] = df['Recession'].shift(-3)
df['Recession_Next_6M'] = df['Recession'].shift(-6)
df['Recession_Next_12M'] = df['Recession'].shift(-12)

# Cut off at last full month
today = datetime(2025, 3, 24)
last_week = today - timedelta(days=7)
last_full_month = last_week.replace(day=1) - timedelta(days=1)
df_full = df.loc[:last_full_month.strftime('%Y-%m-%d')]

# Training data for each horizon
df_train_3m = df_full.dropna(subset=['Recession_Next_3M'])
df_train_6m = df_full.dropna(subset=['Recession_Next_6M'])
df_train_12m = df_full.dropna(subset=['Recession_Next_12M'])

# Features
features = ['GDP_Growth', 'Yield_Spread', 'Unemployment', 'CPI', 'NASDAQ_Return']

# 3-month horizon
X_3m = df_train_3m[features]
y_3m = df_train_3m['Recession_Next_3M']
X_train_3m, X_test_3m, y_train_3m, y_test_3m = train_test_split(X_3m, y_3m, test_size=0.2, random_state=42)
scaler_3m = StandardScaler()
X_train_scaled_3m = scaler_3m.fit_transform(X_train_3m)
X_test_scaled_3m = scaler_3m.transform(X_test_3m)
model_3m = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_3m.fit(X_train_scaled_3m, y_train_3m)

# 6-month horizon
X_6m = df_train_6m[features]
y_6m = df_train_6m['Recession_Next_6M']
X_train_6m, X_test_6m, y_train_6m, y_test_6m = train_test_split(X_6m, y_6m, test_size=0.2, random_state=42)
scaler_6m = StandardScaler()
X_train_scaled_6m = scaler_6m.fit_transform(X_train_6m)
X_test_scaled_6m = scaler_6m.transform(X_test_6m)
model_6m = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_6m.fit(X_train_scaled_6m, y_train_6m)

# 12-month horizon
X_12m = df_train_12m[features]
y_12m = df_train_12m['Recession_Next_12M']
X_train_12m, X_test_12m, y_train_12m, y_test_12m = train_test_split(X_12m, y_12m, test_size=0.2, random_state=42)
scaler_12m = StandardScaler()
X_train_scaled_12m = scaler_12m.fit_transform(X_train_12m)
X_test_scaled_12m = scaler_12m.transform(X_test_12m)
model_12m = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_12m.fit(X_train_scaled_12m, y_train_12m)

# Evaluate models
print("3-Month Horizon Classification Report:")
print(classification_report(y_test_3m, model_3m.predict(X_test_scaled_3m)))
print("\n6-Month Horizon Classification Report:")
print(classification_report(y_test_6m, model_6m.predict(X_test_scaled_6m)))
print("\n12-Month Horizon Classification Report:")
print(classification_report(y_test_12m, model_12m.predict(X_test_scaled_12m)))

# Predict probabilities over the entire dataset
X_full_scaled_3m = scaler_3m.transform(df_full[features])
df_full.loc[:, 'Recession_Prob_3M'] = model_3m.predict_proba(X_full_scaled_3m)[:, 1]
X_full_scaled_6m = scaler_6m.transform(df_full[features])
df_full.loc[:, 'Recession_Prob_6M'] = model_6m.predict_proba(X_full_scaled_6m)[:, 1]
X_full_scaled_12m = scaler_12m.transform(df_full[features])
df_full.loc[:, 'Recession_Prob_12M'] = model_12m.predict_proba(X_full_scaled_12m)[:, 1]

# Predict probabilities for latest data
X_latest = df_full[features].tail(1)
latest_prob_3m = model_3m.predict_proba(scaler_3m.transform(X_latest))[0, 1]
latest_prob_6m = model_6m.predict_proba(scaler_6m.transform(X_latest))[0, 1]
latest_prob_12m = model_12m.predict_proba(scaler_12m.transform(X_latest))[0, 1]

# Display results
print(f"\nPredicted probabilities from {df_full.index[-1]}:")
print(f"  Recession within 3 months: {latest_prob_3m:.2%}")
print(f"  Recession within 6 months: {latest_prob_6m:.2%}")
print(f"  Recession within 12 months: {latest_prob_12m:.2%}")

# Plotting recession probabilities separately for each horizon

# 3-Month Horizon Plot
plt.figure(figsize=(14, 6))
plt.plot(df_full.index, df_full['Recession_Prob_3M'], label='3-Month Probability', color='green')
recession_periods = df_full['Recession'] == 1
plt.fill_between(df_full.index, 0, 1, where=recession_periods, color='gray', alpha=0.3, label='Actual Recessions')
plt.title('3-Month Recession Probability Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1)
plt.scatter(df_full.index[-1], latest_prob_3m, color='green', zorder=5)
plt.text(df_full.index[-1], latest_prob_3m + 0.05, f'{latest_prob_3m:.2%}', color='green')
plt.tight_layout()
plt.show()

# 6-Month Horizon Plot
plt.figure(figsize=(14, 6))
plt.plot(df_full.index, df_full['Recession_Prob_6M'], label='6-Month Probability', color='blue')
plt.fill_between(df_full.index, 0, 1, where=recession_periods, color='gray', alpha=0.3, label='Actual Recessions')
plt.title('6-Month Recession Probability Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1)
plt.scatter(df_full.index[-1], latest_prob_6m, color='blue', zorder=5)
plt.text(df_full.index[-1], latest_prob_6m + 0.05, f'{latest_prob_6m:.2%}', color='blue')
plt.tight_layout()
plt.show()

# 12-Month Horizon Plot
plt.figure(figsize=(14, 6))
plt.plot(df_full.index, df_full['Recession_Prob_12M'], label='12-Month Probability', color='orange')
plt.fill_between(df_full.index, 0, 1, where=recession_periods, color='gray', alpha=0.3, label='Actual Recessions')
plt.title('12-Month Recession Probability Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1)
plt.scatter(df_full.index[-1], latest_prob_12m, color='orange', zorder=5)
plt.text(df_full.index[-1], latest_prob_12m + 0.05, f'{latest_prob_12m:.2%}', color='orange')
plt.tight_layout()
plt.show()

# Plot feature trends with recessions
fig, axes = plt.subplots(len(features), 1, figsize=(14, 10), sharex=True)
for i, feature in enumerate(features):
    axes[i].plot(df_full.index, df_full[feature], label=feature, color='purple')
    axes[i].fill_between(df_full.index, df_full[feature].min(), df_full[feature].max(),
                         where=recession_periods, color='gray', alpha=0.3)
    axes[i].set_ylabel(feature)
    axes[i].legend(loc='upper left')
    axes[i].grid(True, linestyle='--', alpha=0.7)
axes[0].set_title('Economic Indicators with Recession Periods', fontsize=16)
axes[-1].set_xlabel('Date', fontsize=12)
plt.tight_layout()
plt.show()
