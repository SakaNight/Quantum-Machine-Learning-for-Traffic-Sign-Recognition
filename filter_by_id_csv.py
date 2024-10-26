import pandas as pd

# 定义文件路径
input_file_path = '/home/jamie/Works/Waterloo/ECE730/project/data/GTSRB/archive/Train.csv'
output_file_path = 'Filtered_ClassIds_12.csv'

# 指定要提取的多个 ClassId
target_class_ids = [1,2]  # 将多个 ClassId 以列表形式列出

# 读取 CSV 文件并筛选指定 ClassId 的行
data = pd.read_csv(input_file_path)
filtered_data = data[data['ClassId'].isin(target_class_ids)]

# 将筛选后的数据写入新的 CSV 文件
filtered_data.to_csv(output_file_path, index=False)

print(f"筛选后的数据已保存到 {output_file_path}")