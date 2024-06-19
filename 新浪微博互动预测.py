# -*- coding: gbk -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# ��ȡ����
traindata = []  # ѵ������
predictdata = []  # Ԥ������

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

# �ֶ��������
traindata.columns = ["uid", "mid", "time", "forward_count", "comment_count", "like_count", "content"]
predictdata.columns = ["uid", "mid", "time", "content"]

# �������н��м򵥴���ʾ�����ɸ���ʵ�������޸ģ�
traindata['content'] = traindata['content'].apply(lambda x: len(x) if x is not None else 0)
predictdata['content'] = predictdata['content'].apply(lambda x: len(x) if x is not None else 0)

# ѡ��������Ŀ�����
X_train = traindata[["content"]]
y_train = traindata[["forward_count", "comment_count", "like_count"]]

X_test = predictdata[["content"]]

# ����ѵ��������֤��
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ѵ�����ɭ��ģ��
model = RandomForestRegressor()
model.fit(X_train, y_train)

# ����֤���Ͻ���Ԥ��
y_pred_val = model.predict(X_val)

# ����������
mse = mean_squared_error(y_val, y_pred_val)
print("��֤���������:", mse)

# �Բ��Լ�����Ԥ��
y_pred_test = model.predict(X_test)

# ���Ԥ����
for i, pred in enumerate(y_pred_test):
    print(f"forward_count: {int(pred[0])}, comment_count: {int(pred[1])},like_count: {int(pred[2])}")

# ��Ԥ������ predictdata ���������ϳ��µ����ݿ�
result_df = pd.DataFrame({
    'uid': predictdata['uid'],
    'id': predictdata['mid'],
    'forward_count_pred': [int(pred[0]) for pred in y_pred_test],
    'comment_count_pred': [int(pred[1]) for pred in y_pred_test],
    'like_count_pred': [int(pred[2]) for pred in y_pred_test]
})

# �������ݿ򱣴�Ϊ txt �ļ�

result_df.to_csv('D:\\python\\work\\weibo_result_data.txt', sep='\t', index=False)