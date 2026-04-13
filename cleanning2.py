import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('ev_charging_patterns.xlsx')

print(f"initial: {df.shape}")
print("nitial colum name:", df.columns.tolist())

required_columns = [
    'User ID', 'Vehicle Model', 'Battery Capacity (kWh)', 
    'Charging Station ID', 'Charging Station Location',
    'Charging Start Time', 'Charging End Time', 'Energy Consumed (kWh)',
    'Charging Duration (hours)', 'Charging Rate (kW)', 'Charging Cost (USD)',
    'Time of Day', 'Day of Week', 'State of Charge (Start %)',
    'State of Charge (End %)', 'Distance Driven (since last charge) (km)',
    'Temperature (掳C)', 'Vehicle Age (years)', 'Charger Type', 'User Type'
]

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print(f"not found: {missing_cols}")

available_cols = [col for col in required_columns if col in df.columns]
df = df[available_cols].copy()

df['Charging Start Time'] = pd.to_datetime(df['Charging Start Time'], errors='coerce')
df['Charging End Time'] = pd.to_datetime(df['Charging End Time'], errors='coerce')

initial_len = len(df)
df = df.dropna(subset=['Charging Start Time', 'Charging End Time'])
print(f"delete {initial_len - len(df)} colums")

numeric_columns = [
    'Battery Capacity (kWh)', 'Energy Consumed (kWh)', 
    'Charging Rate (kW)', 'Charging Cost (USD)', 
    'State of Charge (Start %)', 'State of Charge (End %)', 
    'Distance Driven (since last charge) (km)',
    'Temperature (掳C)', 'Vehicle Age (years)'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

initial_len = len(df)
df = df.dropna(subset=numeric_columns)
print(f"delete {initial_len - len(df)} colums without values")

categorical_columns = [
    'Vehicle Model', 'Charging Station Location',
    'Time of Day', 'Day of Week', 'Charger Type', 'User Type'
]

for col in categorical_columns:
    df[col] = df[col].astype(str).str.strip()
    df = df[df[col] != 'nan'] 
df['actual_duration_hours'] = (
    (df['Charging End Time'] - df['Charging Start Time']).dt.total_seconds() / 3600
)

df = df[(df['actual_duration_hours'] > 0) & (df['actual_duration_hours'] <= 24)]

df['recalculated_charging_rate_kW'] = df['Energy Consumed (kWh)'] / df['actual_duration_hours']

rate_diff = np.abs(df['Charging Rate (kW)'] - df['recalculated_charging_rate_kW'])
print(f"error: {rate_diff.mean():.2f} kW")


df['C_rate'] = df['recalculated_charging_rate_kW'] / df['Battery Capacity (kWh)']

df['DoD'] = df['State of Charge (End %)'] - df['State of Charge (Start %)']

df['is_cold_weather'] = (df['Temperature (掳C)'] < 0).astype(int)
df['is_hot_weather'] = (df['Temperature (掳C)'] > 35).astype(int)

df['is_fast_charging'] = (df['recalculated_charging_rate_kW'] >= 50).astype(int)

df['is_full_charge'] = (df['State of Charge (End %)'] >= 95).astype(int)

df['energy_per_km'] = df['Energy Consumed (kWh)'] / df['Distance Driven (since last charge) (km)']
df['energy_per_km'] = df['energy_per_km'].replace([np.inf, -np.inf], np.nan).fillna(0)


df['is_full_charge'] = (df['State of Charge (End %)'] >= 95).astype(int)

c_rate_stress = np.where(df['C_rate'] > 1.0, df['C_rate'] - 1.0, 0)

temp_deviation = np.where(
    df['Temperature (掳C)'] < 10,
    10 - df['Temperature (掳C)'],
    np.where(df['Temperature (掳C)'] > 30,
             df['Temperature (掳C)'] - 30, 0)
)

max_temp_dev = 30.0
temp_stress = temp_deviation / max_temp_dev


high_soc_start = np.clip((df['State of Charge (Start %)'] - 80) / 20, 0, 1)  # 80%→0, 100%→1
full_charge_penalty = df['is_full_charge'].astype(float)                    # 0 or 1


soc_stress = 0.7 * high_soc_start + 0.3 * full_charge_penalty

weight_c = 0.5 
weight_temp = 0.3  
weight_soc = 0.2  

df['stress_score'] = (
    weight_c * c_rate_stress +
    weight_temp * temp_stress +
    weight_soc * soc_stress
)

df['stress_level'] = pd.cut(
    df['stress_score'],
    bins=[-np.inf, 0.3, 0.8, np.inf], 
    labels=['Green', 'Yellow', 'Red'],
    include_lowest=True
)
simplified_columns = {
    'User ID': 'user_id',
    'Vehicle Model': 'vehicle_model',
    'Battery Capacity (kWh)': 'battery_capacity',
    'Charging Station ID': 'station_id',
    'Charging Station Location': 'station_location',
    'Charging Start Time': 'start_time',
    'Charging End Time': 'end_time',
    'Energy Consumed (kWh)': 'energy_consumed',
    'Charging Duration (hours)': 'original_duration',
    'Charging Rate (kW)': 'original_rate',
    'Charging Cost (USD)': 'charging_cost',
    'Time of Day': 'time_of_day',
    'Day of Week': 'day_of_week',
    'State of Charge (Start %)': 'soc_start',
    'State of Charge (End %)': 'soc_end',
    'Distance Driven (since last charge) (km)': 'distance_driven',
    'Temperature (掳C)': 'temperature',
    'Vehicle Age (years)': 'vehicle_age',
    'Charger Type': 'charger_type',
    'User Type': 'user_type',
    'actual_duration_hours': 'charging_duration',
    'recalculated_charging_rate_kW': 'charging_rate',
    'C_rate': 'c_rate',
    'DoD': 'dod',
    'stress_score': 'stress_score',
    'stress_level': 'stress_level'
}

df_enhanced = df.rename(columns=simplified_columns)

output_file = 'ev_charging_new.xlsx'
df_enhanced.to_excel(output_file, index=False)
print(f"\nsaved: {output_file}")
print(f"fianlly: {df_enhanced.shape}")

plt.figure(figsize=(8, 5))
df_enhanced['stress_level'].value_counts().plot(kind='bar', color=['green', 'orange', 'red'])
plt.title('Battery pressure rating distribution')
plt.ylabel('frequency')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('stress_level_distribution.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_enhanced, x='temperature', y='c_rate', hue='stress_level', alpha=0.7)
plt.title('C-rate and temperature (accroding to strees level')
plt.xlabel('temperature (°C)')
plt.ylabel('C-rate')
plt.legend(title='stress level')
plt.grid(True)
plt.tight_layout()
plt.savefig('c_rate_vs_temperature.png')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col in zip(axes, ['soc_start', 'soc_end']):
    sns.boxplot(data=df_enhanced, x='stress_level', y=col, ax=ax,
                palette={'Green': 'green', 'Yellow': 'orange', 'Red': 'red'})
    ax.set_title(f'{col.replace("_", " ").title()} by Stress Level')
    ax.set_xlabel('stress level')
plt.tight_layout()
plt.savefig('soc_vs_stress.png')
plt.show()

plt.figure(figsize=(10, 5))
user_stress = df_enhanced.groupby('user_type')['stress_score'].mean().sort_values(ascending=False)
user_stress.plot(kind='bar', color='skyblue')
plt.title('Average battery pressure scores for different user types')
plt.ylabel('Average stress_score')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('user_type_stress.png')
plt.show()

df_enhanced['month'] = df_enhanced['start_time'].dt.month
monthly_stress = df_enhanced.groupby('month')['stress_score'].mean()
plt.figure(figsize=(10, 4))
monthly_stress.plot(marker='o')
plt.title('Monthly average battery pressure trend')
plt.xlabel('Month')
plt.ylabel('Average stress_score')
plt.grid(True)
plt.tight_layout()
plt.savefig('monthly_stress_trend.png')
plt.show()

df_enhanced['hour'] = df_enhanced['start_time'].dt.hour
plt.figure(figsize=(12, 5))
sns.boxplot(data=df_enhanced, x='hour', y='stress_score', color='lightcoral')
plt.title('The distribution of battery pressure at different times of the day')
plt.xlabel('hours (0-23)')
plt.ylabel('stress_score')
plt.tight_layout()
plt.savefig('hourly_stress.png')
plt.show()

ct = pd.crosstab(df_enhanced['is_fast_charging'], df_enhanced['stress_level'], normalize='index')
ct.plot(kind='bar', stacked=True, color=['green', 'orange', 'red'], figsize=(8, 5))
plt.title('The proportion of pressure levels among fast charging users')
plt.xlabel('Is it fast charging? (0= No, 1= Yes)')
plt.ylabel('proportion')
plt.xticks(rotation=0)
plt.legend(title='stress level')
plt.tight_layout()
plt.savefig('fast_charge_stress_share.png')
plt.show()

red_charges = df_enhanced[df_enhanced['stress_level'] == 'Red']
if not red_charges.empty:
    top_red_stations = red_charges['station_location'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    top_red_stations.plot(kind='barh', color='crimson')
    plt.title('The Top 10 locations with the most high-pressure (Red) charging incidents')
    plt.xlabel('Red----Number of events')
    plt.ylabel('Location of the charging station')
    plt.tight_layout()
    plt.savefig('high_risk_stations.png')
    plt.show()
else:
    print("no red")

top_models = df_enhanced['vehicle_model'].value_counts().index[:10]
df_top = df_enhanced[df_enhanced['vehicle_model'].isin(top_models)]
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_top, x='vehicle_model', y='stress_score')
plt.xticks(rotation=45)
plt.title('Top 10 Battery pressure distribution of the vehicle model')
plt.tight_layout()
plt.savefig('vehicle_model_stress.png')
plt.show()