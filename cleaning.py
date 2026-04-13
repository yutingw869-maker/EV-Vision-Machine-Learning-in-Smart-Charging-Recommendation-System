import pandas as pd
import numpy as np

# 1. 加载数据
df = pd.read_excel('ev_charging_patterns.xlsx')

# 2. 查看原始数据信息
print("原始数据形状:", df.shape)
print("\n原始数据列名:")
print(df.columns.tolist())
print("\n原始数据前5行:")
print(df.head())

# 3. 数据清洗
# 3.1 定义需要保留的列（根据提供的列名清单）
required_columns = [
    'User ID', 'Vehicle Model', 'Battery Capacity (kWh)', 
    'Charging Station ID', 'Charging Station Location',
    'Charging Start Time', 'Charging End Time', 'Energy Consumed (kWh)',
    'Charging Duration (hours)', 'Charging Rate (kW)', 'Charging Cost (USD)',
    'Time of Day', 'Day of Week', 'State of Charge (Start %)',
    'State of Charge (End %)', 'Distance Driven (since last charge) (km)',
    'Temperature (掳C)', 'Vehicle Age (years)', 'Charger Type', 'User Type'
]

# 保留需要的列，删除其他列
df = df[required_columns].copy()

# 3.2 处理时间格式
df['Charging Start Time'] = pd.to_datetime(df['Charging Start Time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
df['Charging End Time'] = pd.to_datetime(df['Charging End Time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')

# 3.3 定义数值列和分类列
numeric_columns = [
    'Battery Capacity (kWh)', 'Energy Consumed (kWh)', 'Charging Duration (hours)',
    'Charging Rate (kW)', 'Charging Cost (USD)', 'State of Charge (Start %)',
    'State of Charge (End %)', 'Distance Driven (since last charge) (km)',
    'Temperature (掳C)', 'Vehicle Age (years)'
]

categorical_columns = [
    'Vehicle Model', 'Charging Station Location', 'Time of Day',
    'Day of Week', 'Charger Type', 'User Type'
]

# 3.4 清洗数值列：去除非数值型数据，处理空值
for col in numeric_columns:
    # 将列转换为数值类型，无法转换的设为NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除该列中空值的行
    initial_count = len(df)
    df = df.dropna(subset=[col])
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"从 {col} 中删除了 {removed_count} 行空值或非数值数据")

# 3.5 清洗分类列：处理空值
for col in categorical_columns:
    # 删除该列中空值的行
    initial_count = len(df)
    df = df.dropna(subset=[col])
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"从 {col} 中删除了 {removed_count} 行空值")

# 3.6 确保分类列是字符串类型
for col in categorical_columns:
    df[col] = df[col].astype(str).str.strip()

# 3.7 删除完全重复的行
initial_count = len(df)
df = df.drop_duplicates()
removed_count = initial_count - len(df)
if removed_count > 0:
    print(f"删除了 {removed_count} 行完全重复的数据")

# 4. 查看清洗后的数据信息
print(f"\n清洗后数据形状:", df.shape)
print("\n清洗后数据信息:")
print(df.info())

# 5. 分类变量统计
categorical_stats = {}
print("\n" + "="*50)
print("分类变量统计")
print("="*50)

for col in categorical_columns:
    print(f"\n【{col}】")
    count_df = df[col].value_counts().reset_index()
    count_df.columns = [col, 'count']
    count_df = count_df.sort_values('count', ascending=False)
    
    categorical_stats[col] = count_df
    print(count_df.to_string(index=False))

# 6. 简化列名并保存清洗后的数据
simplified_columns = {
    'User ID': 'user_id',
    'Vehicle Model': 'vehicle_model',
    'Battery Capacity (kWh)': 'battery_capacity',
    'Charging Station ID': 'station_id',
    'Charging Station Location': 'station_location',
    'Charging Start Time': 'start_time',
    'Charging End Time': 'end_time',
    'Energy Consumed (kWh)': 'energy_consumed',
    'Charging Duration (hours)': 'charging_duration',
    'Charging Rate (kW)': 'charging_rate',
    'Charging Cost (USD)': 'charging_cost',
    'Time of Day': 'time_of_day',
    'Day of Week': 'day_of_week',
    'State of Charge (Start %)': 'soc_start',
    'State of Charge (End %)': 'soc_end',
    'Distance Driven (since last charge) (km)': 'distance_driven',
    'Temperature (掳C)': 'temperature',
    'Vehicle Age (years)': 'vehicle_age',
    'Charger Type': 'charger_type',
    'User Type': 'user_type'
}

df_cleaned = df.rename(columns=simplified_columns)

# 保存清洗后的数据
df_cleaned.to_excel('ev_charging_cleaned.xlsx', index=False)
print(f"\n清洗后的数据已保存到: ev_charging_cleaned.xlsx")

# 7. 保存分类变量统计结果
with pd.ExcelWriter('ev_categorical_stats.xlsx', engine='openpyxl') as writer:
    for col, stats_df in categorical_stats.items():
        # 简化工作表名称
        sheet_name = col.replace(' ', '_').lower()[:31]  # 限制工作表名称长度
        stats_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"分类变量统计结果已保存到: ev_categorical_stats.xlsx")

# 8. 输出清洗摘要
print(f"\n" + "="*50)
print("数据清洗摘要")
print("="*50)
print(f"原始数据行数: {initial_count}")
print(f"清洗后数据行数: {len(df_cleaned)}")
print(f"保留率: {len(df_cleaned)/initial_count*100:.2f}%")
print(f"\n数值列数量: {len(numeric_columns)}")
print(f"分类列数量: {len(categorical_columns)}")
print(f"总列数: {len(df_cleaned.columns)}")

# 显示每个分类变量的唯一值数量
print(f"\n各分类变量唯一值数量:")
for col in categorical_columns:
    simplified_col = simplified_columns[col]
    unique_count = df_cleaned[simplified_col].nunique()
    print(f"  {col}: {unique_count}")