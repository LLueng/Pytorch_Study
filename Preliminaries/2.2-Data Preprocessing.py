import os
import pandas as pd
import torch

# 2.2、数据预处理
# 使用pandas预处理原始数据
# 2.2.1、读取数据集
# 2.2.1.1、我们首先创建一个人工数据集
# 并存储在CSV（逗号分隔值）文件 ../data/house_tiny.csv中。
# 以其他格式存储的数据也可以通过类似的方式进行处理。
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
# 2.2.1.2、要从创建的CSV文件中加载原始数据集，我们导入pandas包并调用read_csv函数。该数据集有四行三列。
# 其中每行描述了房间数量（“NumRooms”）、巷子类型（“Alley”）和房屋价格（“Price”）。
data = pd.read_csv(data_file)
# 打印结果：
#    NumRooms Alley   Price
# 0       NaN  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       NaN   NaN  140000
print(data)

# 2.2.2、处理缺失值
# “NaN”项代表缺失值。 为了处理缺失的数据，典型的方法包括插值法和删除法， 
# 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。
# 在这里，我们将考虑插值法。通过位置索引iloc，我们将data分成inputs和outputs， 
# 其中前者为data的前两列，而后者为data的最后一列。 对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
# 提取输入和输出
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 将数据转换为数值类型
inputs = inputs.apply(pd.to_numeric, errors='coerce')
# 用 NumRooms 列的平均值填充该列的 NaN 值
data['NumRooms'] = data['NumRooms'].fillna(data['NumRooms'].mean())
# 打印结果：
#    NumRooms Alley
# 0       3.0  Pave
# 1       2.0   NaN
# 2       4.0   NaN
# 3       3.0   NaN
print(inputs)

# 2.2.3、对于inputs中的类别值或离散值，我们将“NaN”视为一个类别。 
# 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， 
# pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。 
# 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 
# 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。
# 使用 get_dummies 函数进行转换，设置 dummy_na=True
# 对 Alley 列进行独热编码，同时保留 NumRooms 列
inputs = pd.get_dummies(data, columns=['Alley'], dummy_na=True)
# 重命名列名，使其符合你的要求
inputs.rename(columns={'Alley_Pave': 'Alley_Pave', 'Alley_nan': 'Alley_nan'}, inplace=True)
# 确保 Alley_Pave 和 Alley_nan 列的值符合要求
inputs['Alley_Pave'] = (data['Alley'] == 'Pave').astype(int)
inputs['Alley_nan'] = data['Alley'].isna().astype(int)
# 打印结果：
#    NumRooms  Alley_Pave  Alley_nan
# 0       3.0           1          0
# 1       2.0           0          1
# 2       4.0           0          1
# 3       3.0           0          1
print(inputs)

# 2.2.4、转换为张量格式
# 现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
# 打印结果：
# (tensor([[3., 1., 0.],
#          [2., 0., 1.],
#          [4., 0., 1.],
#          [3., 0., 1.]], dtype=torch.float64),
#  tensor([127500., 106000., 178100., 140000.], dtype=torch.float64))
print(X, y)

# 2.2.5、总结
# pandas软件包是Python中常用的数据分析工具中，pandas可以与张量兼容。
# 用pandas处理缺失的数据时，我们可根据情况选择用插值法和删除法。
