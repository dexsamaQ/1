# -*- coding: gbk -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# 读取数据
traindata = []  # 训练数据
predictdata = []  # 预测数据

with open('D:\python\work\weibo_train_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            row = line.strip().split('\t')
            traindata.append(row)
        except Exception as e:
            print(f"Error parsing line: {line}")

with open('D:\python\work\weibo_predict_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            row = line.strip().split('\t')
            predictdata.append(row)
        except Exception as e:
            print(f"Error parsing line: {line}")

traindata = pd.DataFrame(traindata)
predictdata = pd.DataFrame(predictdata)

# 手动添加列名
traindata.columns = ["uid", "mid", "time", "forward_count", "comment_count", "like_count", "content"]
predictdata.columns = ["uid", "mid", "time", "content"]

# 对内容列进行简单处理（示例，可根据实际需求修改）
traindata['content'] = traindata['content'].apply(lambda x: len(x) if x is not None else 0)
predictdata['content'] = predictdata['content'].apply(lambda x: len(x) if x is not None else 0)

# 选择特征和目标变量
X_train = traindata[["content"]]
y_train = traindata[["forward_count", "comment_count", "like_count"]]

X_test = predictdata[["content"]]

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 在验证集上进行预测
y_pred_val = model.predict(X_val)

# 计算均方误差
mse = mean_squared_error(y_val, y_pred_val)
print("验证集均方误差:", mse)

# 对测试集进行预测
y_pred_test = model.predict(X_test)

# 输出预测结果
for i, pred in enumerate(y_pred_test):
    print(f"forward_count: {int(pred[0])}, comment_count: {int(pred[1])},like_count: {int(pred[2])}")

# 将预测结果与 predictdata 的相关列组合成新的数据框
result_df = pd.DataFrame({
    'uid': predictdata['uid'],
    'id': predictdata['mid'],
    'forward_count_pred': [int(pred[0]) for pred in y_pred_test],
    'comment_count_pred': [int(pred[1]) for pred in y_pred_test],
    'like_count_pred': [int(pred[2]) for pred in y_pred_test]
})

# 将新数据框保存为 txt 文件

result_df.to_csv('D:\\python\\work\\weibo_result_data.txt', sep='\t', index=False)