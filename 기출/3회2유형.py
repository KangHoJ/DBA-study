import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score , accuracy_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
pd.options.display.max_columns=900

# 예측값 TravelInsurance (여행보험 패키지 구매여부)
train = pd.read_csv('./t2-1-train.csv') 
test = pd.read_csv('./t2-1-test.csv')   
# print(train.head())
print('초기load shape : ',train.shape , test.shape)


# ################### EDA ###############################
# 결측값 확인
# print(train.isnull().sum()) 
train['AnnualIncome'] = train['AnnualIncome'].fillna(train['AnnualIncome'].median())
test['AnnualIncome'] = test['AnnualIncome'].fillna(test['AnnualIncome'].median())
# print(train.isnull().sum()) 

# 유형별로 나누기 
# print(train.info())
cat_col  = train.select_dtypes(include='object').columns # ['Employment Type', 'GraduateOrNot', 'FrequentFlyer','EverTravelledAbroad']
num_col  = train.select_dtypes(exclude='object').columns # ['id', 'Age', 'AnnualIncome', 'FamilyMembers', 'ChronicDiseases','TravelInsurance']
# print(cat_col , num_col)

# 이상치 확인
# print(train.describe()) # 딱히 없어보임

# 상관관계 확인
# print(train[num_col].corr())

# drop
train = train.drop(['Employment Type','ChronicDiseases'],axis=1)
test = test.drop(['Employment Type','ChronicDiseases'],axis=1)
print('드랍후 shape : ',train.shape , test.shape)

test_id = test['id']

# 1차 분리
x_train = train.drop('TravelInsurance',axis=1)
y_train = train['TravelInsurance']
x_test = test

print('1차 분리후 shape',x_train.shape,y_train.shape,x_test.shape)

# 인코딩 및 스케일링 
cat_col = ['GraduateOrNot', 'FrequentFlyer','EverTravelledAbroad']
num_col  = ['id', 'Age', 'AnnualIncome', 'FamilyMembers']

x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)
print('인코딩후 shape',x_train.shape,y_train.shape,x_test.shape) # test가 하나 더 많음
x_train = x_train.reindex(columns=x_test.columns,fill_value=0) # 처리 
print('인코딩 교정후 shape',x_train.shape,y_train.shape,x_test.shape) # 처리후 shape 


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# x_train[num_col] = scaler.fit_transform(x_train[num_col])
# x_test[num_col] = scaler.transform(x_test[num_col])

# from sklearn.preprocessing import LabelEncoder
# for i in cat_col:
#     encoder = LabelEncoder()
#     x_train[i] = encoder.fit_transform(x_train[i])
#     x_test[i] = encoder.transform(x_test[i])

# 분리
x_tr , x_val , y_tr , y_val = train_test_split(x_train,y_train,test_size=0.3,stratify=y_train,random_state=2023)
print('2차분리후' , x_tr.shape , x_val.shape , y_tr.shape , y_val.shape)


# print(lightgbm.__all__)
model = RandomForestClassifier(max_depth=5,random_state=2023,n_estimators=150)
model2 = LGBMClassifier()

model.fit(x_train,y_train)
tr_pred = model.predict_proba(x_train)
te_pred = model.predict_proba(x_val)
from sklearn.metrics import roc_auc_score
roc = roc_auc_score(y_tr, tr_pred[:,1])
roc2 = roc_auc_score(y_val, te_pred[:,1])
print(roc , roc2)

# result = model.predict_proba(x_test)
# result_df = pd.DataFrame({'id':test['id'],'TravelInsurance':result[:,1]})
# result_df.to_csv('3_submission.csv',index=False)

# d_test = pd.read_csv('3_submission.csv')
# print(d_test)
