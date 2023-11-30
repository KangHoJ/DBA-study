import pandas as pd 
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

train = pd.read_csv('./tr.csv')
x_test = pd.read_csv('./te.csv')

# print(train.head())

x_train = train.drop('Segmentation',axis=1)
y_train = train['Segmentation']

print(x_train.shape,x_test.shape,y_train.shape)
# print(x_train.info())

num_col = x_train.select_dtypes(exclude='object').columns
cat_col = x_train.select_dtypes(include='object').columns
print(num_col) # ['ID', 'Age', 'Work_Experience', 'Family_Size']
print(cat_col) # ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score','Var_1']
# print(x_train['Var_1'].value_counts())
# import sklearn.metrics
# print(sklearn.metrics.__all__)

x_test_id = x_test['ID']

X = pd.get_dummies(x_train)
Test = pd.get_dummies(x_test)
Y = y_train

x_tr , x_val , y_tr , y_val = train_test_split(X.values,Y.values,test_size=0.1,random_state=100)
print(x_tr.shape , x_val.shape , y_tr.shape , y_val.shape)
# print(Y)

model = GradientBoostingClassifier()
# model2 = XGBClassifier()

model.fit(x_tr,y_tr)
predict = model.predict(x_val)
score = f1_score(y_val,predict,average='macro')
print(score)

pred = model.predict(Test)
output = pd.DataFrame({'ID':x_test_id,'Segmentation':pred}).to_csv('submission.csv',index=False)
print(output)