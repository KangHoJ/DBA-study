import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

x_train = pd.read_csv("./t2-2-X_train.csv")
y_train = pd.read_csv("./t2-2-y_train.csv")
x_test = pd.read_csv("./t2-2-X_test.csv")

print(x_train.shape , y_train.shape , x_test.shape)
train = pd.merge(x_train,y_train,how='left',on='Serial No.')
print(train.shape , x_test.shape)
########################################################
# 타겟 확인
# print(train.columns)
# print(train['Chance of Admit '].describe())

# 데이터 분리
cat_col = train.select_dtypes(include='object').columns
num_col = train.select_dtypes(exclude='object').columns

# 결측값 확인
# print(train.isnull().sum() , x_test.isnull().sum()) # 결측값 x

# 이상치 확인
# print(train.describe())

# 상관관계 확인
# print(train[num_col].corr())

# 필요없는 컬럼 drop
train = train.drop('Serial No.',axis=1)
x_test = x_test.drop('Serial No.',axis=1)
print('drop후 shape',train.shape , x_test.shape)

# 다시 분리
x_train = train.drop('Chance of Admit ',axis=1)
y_train = train['Chance of Admit ']
x_test = x_test
print('1차분리후 shape',x_train.shape , y_train.shape , x_test.shape )

# 인코딩
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)
print('인코딩후 shape',x_train.shape , y_train.shape , x_test.shape )

# 스케일러
num_col = x_train.select_dtypes(exclude='object').columns
sc = StandardScaler()
for i in num_col:
    x_train[i] =sc.fit_transform(x_train[[i]])
    x_test[i] =sc.transform(x_test[[i]])


x_tees_id = x_test.index
# 평가를 위한 분리 
x_tr , x_te , y_tr , y_te = train_test_split(x_train,y_train,test_size=0.05,random_state=2023)
print('2차분리후 shape',x_tr.shape , x_te.shape , y_tr.shape , y_te.shape)

model = RandomForestRegressor(random_state=2023,max_depth=6)
model2 = LGBMRegressor(random_state=2023)
model3 = GradientBoostingRegressor(random_state=2023)

model.fit(x_train,y_train)
pred_tr = model.predict(x_tr)
pred_te = model.predict(x_te)

score_tr = r2_score(y_tr,pred_tr)
score_te = r2_score(y_te,pred_te)


print('train score :',score_tr)
print('test score :',score_te)

target = model.predict(x_test)
# print(target)
df = pd.DataFrame({'id':x_tees_id,'result':target})
print(df)
