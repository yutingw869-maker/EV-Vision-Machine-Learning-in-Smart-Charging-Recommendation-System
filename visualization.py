import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from datetime import datetime

# --------------------------
# 1. Data Loading and Preprocessing
# --------------------------
# Load cleaned data
df = pd.read_excel('ev_charging_cleaned.xlsx')

# Basic data inspection
print("Data shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Convert start_time to datetime and extract time features
df['start_time'] = pd.to_datetime(df['start_time'])
df['hour'] = df['start_time'].dt.hour
df['day_of_week'] = df['start_time'].dt.day_name()  # English day names
df['date'] = df['start_time'].dt.date

# Define charging time periods
def get_time_period(hour):
    if 6 <= hour < 12:
        return 'Morning (6-12)'
    elif 12 <= hour < 18:
        return 'Afternoon (12-18)'
    elif 18 <= hour < 24:
        return 'Evening (18-24)'
    else:
        return 'Night (0-6)'

df['time_period'] = df['hour'].apply(get_time_period)

# Calculate battery utilization rate
df['battery_utilization'] = (df['energy_consumed'] / df['battery_capacity']) * 100

# Round vehicle age to integer for analysis
df['vehicle_age_int'] = df['vehicle_age'].round().astype(int)

# --------------------------
# 2. User-Side Analysis Charts
# --------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('User-Side Charging Behavior Analysis', fontsize=16, fontweight='bold')

# 2.1 Average Charging Duration by Vehicle Model
vehicle_duration = df.groupby('vehicle_model')['charging_duration'].agg(['mean', 'count']).reset_index()
vehicle_duration = vehicle_duration.sort_values('mean', ascending=False)

ax1 = axes[0, 0]
bars = ax1.bar(vehicle_duration['vehicle_model'], vehicle_duration['mean'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
ax1.set_title('Average Charging Duration by Vehicle Model', fontsize=12, fontweight='bold')
ax1.set_xlabel('Vehicle Model')
ax1.set_ylabel('Average Charging Duration (hours)')
ax1.tick_params(axis='x', rotation=45)

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10)

# 2.2 Charging Time Period Preference by Vehicle Model
vehicle_period = pd.crosstab(df['vehicle_model'], df['time_period'], normalize='index') * 100
vehicle_period.plot(kind='bar', stacked=True, ax=axes[0, 1], 
                    color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
axes[0, 1].set_title('Charging Time Period Distribution by Vehicle Model', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Vehicle Model')
axes[0, 1].set_ylabel('Percentage (%)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].legend(title='Time Period', bbox_to_anchor=(1.05, 1), loc='upper left')

# 2.3 Charging Frequency by User Type
user_charges = df.groupby('user_type')['user_id'].count().reset_index()
user_charges.columns = ['User Type', 'Charging Count']

ax3 = axes[1, 0]
bars = ax3.bar(user_charges['User Type'], user_charges['Charging Count'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax3.set_title('Charging Frequency by User Type', fontsize=12, fontweight='bold')
ax3.set_xlabel('User Type')
ax3.set_ylabel('Number of Charging Sessions')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

# 2.4 Average Charging Duration by User Type
user_duration = df.groupby('user_type')['charging_duration'].agg(['mean', 'std']).reset_index()

ax4 = axes[1, 1]
x = range(len(user_duration))
bars = ax4.bar(x, user_duration['mean'], yerr=user_duration['std'], 
               capsize=5, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax4.set_title('Average Charging Duration by User Type', fontsize=12, fontweight='bold')
ax4.set_xlabel('User Type')
ax4.set_ylabel('Average Charging Duration (hours)')
ax4.set_xticks(x)
ax4.set_xticklabels(user_duration['user_type'])

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('user_side_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("User-side analysis chart saved")

# --------------------------
# 3. Spatiotemporal Analysis Charts
# --------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Spatiotemporal Charging Behavior Analysis', fontsize=16, fontweight='bold')

# 3.1 Daily Charging Peak Distribution
hourly_charges = df.groupby('hour')['user_id'].count().reset_index()
hourly_charges.columns = ['Hour', 'Charging Count']

ax1 = axes[0, 0]
ax1.plot(hourly_charges['Hour'], hourly_charges['Charging Count'], 
         marker='o', linewidth=2, markersize=6, color='#FF6B6B')
ax1.fill_between(hourly_charges['Hour'], hourly_charges['Charging Count'], 
                 alpha=0.3, color='#FF6B6B')
ax1.set_title('Intraday Charging Peak Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Number of Charging Sessions')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, 24, 2))

max_hour = hourly_charges.loc[hourly_charges['Charging Count'].idxmax()]
ax1.annotate(f'Peak: {max_hour["Hour"]}:00\n{max_hour["Charging Count"]} sessions',
             xy=(max_hour['Hour'], max_hour['Charging Count']),
             xytext=(max_hour['Hour']+2, max_hour['Charging Count']+20),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red', fontweight='bold')

# 3.2 Weekly Charging Distribution
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_charges = df.groupby('day_of_week')['user_id'].count().reindex(weekday_order).reset_index()
weekday_charges.columns = ['Day of Week', 'Charging Count']

ax2 = axes[0, 1]
bars = ax2.bar(weekday_charges['Day of Week'], weekday_charges['Charging Count'], 
               color=['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'])
ax2.set_title('Weekly Charging Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Day of Week')
ax2.set_ylabel('Number of Charging Sessions')
ax2.tick_params(axis='x', rotation=45)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

# 3.3 Regional Charging Volume Bar Chart (Replaces empty heatmap)
location_energy = df.groupby('station_location')['energy_consumed'].sum().reset_index()
location_energy = location_energy.sort_values('energy_consumed', ascending=False).head(10)  # Top 10 locations

ax3 = axes[1, 0]
bars = ax3.bar(location_energy['station_location'], location_energy['energy_consumed'], color='#45B7D1')
ax3.set_title('Total Charging Energy by Location (Top 10)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Charging Station Location')
ax3.set_ylabel('Total Energy Consumed (kWh)')
ax3.tick_params(axis='x', rotation=45)

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{int(height)}', ha='center', va='bottom', fontsize=9)

# 3.4 Daily Charging Energy Trend
daily_energy = df.groupby('date')['energy_consumed'].sum().reset_index()
daily_energy['date'] = pd.to_datetime(daily_energy['date'])

ax4 = axes[1, 1]
ax4.plot(daily_energy['date'], daily_energy['energy_consumed'], 
         marker='o', linewidth=2, markersize=4, color='#45B7D1')
ax4.set_title('Daily Charging Energy Trend', fontsize=12, fontweight='bold')
ax4.set_xlabel('Date')
ax4.set_ylabel('Total Energy Consumed (kWh)')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('spatiotemporal_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Spatiotemporal analysis chart saved")

# --------------------------
# 4. Environmental Impact Analysis Charts
# --------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Environmental Impact on Charging Behavior', fontsize=16, fontweight='bold')

# 4.1 Temperature vs Charging Rate
ax1 = axes[0, 0]
scatter = ax1.scatter(df['temperature'], df['charging_rate'], 
                     alpha=0.6, c=df['energy_consumed'], cmap='viridis', s=30)
ax1.set_title('Temperature vs Charging Rate', fontsize=12, fontweight='bold')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Charging Rate (kW)')
ax1.grid(True, alpha=0.3)

cbar1 = plt.colorbar(scatter, ax=ax1)
cbar1.set_label('Energy Consumed (kWh)')

z = np.polyfit(df['temperature'], df['charging_rate'], 1)
p = np.poly1d(z)
ax1.plot(df['temperature'], p(df['temperature']), "r--", alpha=0.8, linewidth=2)

# 4.2 Average Charging Duration by Temperature Range
df['temp_range'] = pd.cut(df['temperature'], bins=[-10, 0, 10, 20, 30, 40], 
                         labels=['<0°C', '0-10°C', '10-20°C', '20-30°C', '>30°C'])
temp_duration = df.groupby('temp_range')['charging_duration'].mean().reset_index()

ax2 = axes[0, 1]
bars = ax2.bar(temp_duration['temp_range'], temp_duration['charging_duration'], 
               color=['#3498DB', '#2ECC71', '#F1C40F', '#E67E22', '#E74C3C'])
ax2.set_title('Average Charging Duration by Temperature Range', fontsize=12, fontweight='bold')
ax2.set_xlabel('Temperature Range')
ax2.set_ylabel('Average Charging Duration (hours)')

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10)

# 4.3 Vehicle Age vs Battery Utilization
ax3 = axes[1, 0]
scatter = ax3.scatter(df['vehicle_age_int'], df['battery_utilization'], 
                     alpha=0.6, c=df['vehicle_model'].astype('category').cat.codes, 
                     cmap='tab10', s=30)
ax3.set_title('Vehicle Age vs Battery Utilization Rate', fontsize=12, fontweight='bold')
ax3.set_xlabel('Vehicle Age (Years)')
ax3.set_ylabel('Battery Utilization Rate (%)')
ax3.grid(True, alpha=0.3)

z = np.polyfit(df['vehicle_age_int'], df['battery_utilization'], 1)
p = np.poly1d(z)
ax3.plot(df['vehicle_age_int'], p(df['vehicle_age_int']), "r--", alpha=0.8, linewidth=2)

# 4.4 Average Charging Rate by Integer Vehicle Age
age_rate = df.groupby('vehicle_age_int')['charging_rate'].agg(['mean', 'std']).reset_index()

ax4 = axes[1, 1]
bars = ax4.bar(age_rate['vehicle_age_int'], age_rate['mean'], yerr=age_rate['std'], 
               capsize=5, color='#9B59B6')
ax4.set_title('Average Charging Rate by Vehicle Age', fontsize=12, fontweight='bold')
ax4.set_xlabel('Vehicle Age (Years)')
ax4.set_ylabel('Average Charging Rate (kW)')
ax4.set_xticks(age_rate['vehicle_age_int'])

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('environmental_impact_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Environmental impact analysis chart saved")

# --------------------------
# 5. User Charging Profile Radar Chart (Fixed Normalization)
# --------------------------
# Calculate user type features
# Recalculate Fast Charger Ratio with the correct label
user_features = df.groupby('user_type').agg({
    'charger_type': lambda x: (x == 'DC Fast Charger').mean() * 100,
    # Keep other metrics as before
    'charging_duration': 'mean',
    'energy_consumed': 'mean',
    'charging_rate': 'mean',
    'battery_utilization': 'mean',
    'time_period': lambda x: (x == 'Night (0-6)').mean() * 100
}).reset_index()

# Features to normalize
features_to_normalize = ['charging_duration', 'energy_consumed', 'charging_rate', 
                        'battery_utilization', 'charger_type', 'time_period']

# Normalize features to [0, 1]
user_features_normalized = user_features.copy()
for feature in features_to_normalize:
    max_val = user_features[feature].max()
    min_val = user_features[feature].min()
    # Avoid division by zero
    if max_val > min_val:
        user_features_normalized[feature] = (user_features[feature] - min_val) / (max_val - min_val)
    else:
        user_features_normalized[feature] = 0

# Radar chart setup
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, polar=True)
# Drop user types with no valid data
user_features = user_features.dropna()

# Proceed to plot only valid user types
# Feature labels and angles
# Normalize features (same as before, but with updated user_features)
features_to_normalize = ['charging_duration', 'energy_consumed', 'charging_rate', 
                        'battery_utilization', 'charger_type', 'time_period']
for feat in features_to_normalize:
    max_val = user_features[feat].max()
    min_val = user_features[feat].min()
    user_features[feat] = (user_features[feat] - min_val) / (max_val - min_val)

# Plot the radar chart
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, polar=True)

feature_labels = ['Avg. Charging Duration', 'Avg. Energy Consumed', 'Avg. Charging Rate', 
                 'Battery Utilization', 'Fast Charger Ratio', 'Night Charging Ratio']
angles = np.linspace(0, 2 * np.pi, len(feature_labels), endpoint=False).tolist()
angles += angles[:1]

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
user_types = user_features['user_type'].unique()

for i, utype in enumerate(user_types):
    data = user_features[user_features['user_type'] == utype][features_to_normalize].values.flatten().tolist()
    data += data[:1]
    ax.plot(angles, data, 'o-', linewidth=2, label=utype, color=colors[i])
    ax.fill(angles, data, alpha=0.25, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(feature_labels, fontsize=11)
ax.set_yticklabels([])
ax.grid(True)

plt.title('Charging Behavior Profiles by User Type', size=16, fontweight='bold', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.savefig('corrected_radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

print("User charging profile radar chart saved")

# --------------------------
# 6. Interactive Regional Charging Heatmap (Plotly)
# --------------------------
# Check if location coordinates exist, otherwise use simulated data
if 'latitude' not in df.columns or 'longitude' not in df.columns:
    print("No geographic coordinates found. Generating simulated coordinates for demonstration.")
    locations = df['station_location'].unique()
    np.random.seed(42)
    latitudes = np.random.uniform(30, 32, len(locations))
    longitudes = np.random.uniform(120, 122, len(locations))
    loc_coords = dict(zip(locations, zip(latitudes, longitudes)))
    
    df['latitude'] = df['station_location'].map(lambda x: loc_coords[x][0])
    df['longitude'] = df['station_location'].map(lambda x: loc_coords[x][1])

# Calculate total energy per location
location_energy = df.groupby('station_location').agg({
    'energy_consumed': 'sum',
    'latitude': 'first',
    'longitude': 'first'
}).reset_index()

# Create interactive heatmap
fig_geo = px.scatter_mapbox(
    location_energy,
    lat="latitude",
    lon="longitude",
    size="energy_consumed",
    color="energy_consumed",
    color_continuous_scale=px.colors.cyclical.IceFire,
    size_max=50,
    zoom=10,
    mapbox_style="carto-positron",
    title="Regional Charging Energy Distribution",
    labels={'energy_consumed': 'Total Energy (kWh)'}
)
fig_geo.write_html('regional_charging_heatmap.html')

print("Interactive regional charging heatmap saved as HTML")

print("\nAll visualizations generated successfully!")
print("Generated files:")
print("1. user_side_analysis.png")
print("2. spatiotemporal_analysis.png") 
print("3. environmental_impact_analysis.png")
print("4. user_charging_profile_radar.png")
print("5. regional_charging_heatmap.html")