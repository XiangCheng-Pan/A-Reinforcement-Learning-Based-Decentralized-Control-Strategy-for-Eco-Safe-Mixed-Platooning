import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 指定Excel文件路径
file_path = 'D:\Doctoral Studies\Project_Studies\PycharmProjects\PytorchPractice\Practice-CommPPO-for-platoon-by-pytorch-main\output\easy-mode-Right.xlsx'  # 替换为你的Excel文件路径
df = pd.read_excel(file_path, sheet_name='Sheet1', usecols=['Reward mean', 'Reward std'])
print(df.columns)
y_mean = np.array(df['Reward mean'])
y_std = np.array(df['Reward std'])
y_max = y_mean + y_std * 0.95
y_min = y_mean - y_std * 0.95
x = np.arange(len(y_mean))
fig = plt.figure(1)
plt.plot(x, y_mean, label='method1', color='#e75840')
plt.fill_between(x, y_max, y_min, alpha=0.6, facecolor='#e75840')
plt.xlabel('Training epochs')
plt.ylabel('Average training return')
plt.legend()
plt.grid(True)
plt.show()
