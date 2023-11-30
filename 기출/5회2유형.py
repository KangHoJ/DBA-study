import pandas as pd
import numpy as np

train = pd.read_csv('./5.train.csv')
test = pd.read_csv('./5.test.csv')
print('import후',train.shape , test.shape)
# print(train.head())

cat_col = train.select_dtypes(include='object').columns # ['model', 'transmission', 'fuelType']
num_col = train.select_dtypes(exclude='object').columns # ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
# print(cat_col) 
# print(num_col)

# print(train[num_col].corr())
# print(train.describe())

# 전처리
# train = train[~(train['year']>2023)]

# Q3 = train['mileage'].quantile(0.75)
# Q1 = train['mileage'].quantile(0.25)
# IQR = Q3 - Q1
# out1 = train['mileage']<Q1-1.5*IQR 
# out2 = train['mileage']>Q3+1.5*IQR 

# train = train[~((out1) | (out2))]  # 이상치제거 
# print(train.describe())


x_train = train.drop('price',axis=1)
y_train = train['price']
x_test = test 
print('첫 분리후 ' ,x_train.shape,y_train.shape,x_test.shape)
# print(x_train.isnull().sum())

x_test_id = x_test.index

# 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scale_col  = x_train.select_dtypes(exclude=object).columns
x_train[scale_col] = scaler.fit_transform(x_train[scale_col])
x_test[scale_col] = scaler.transform(x_test[scale_col])

## 인코딩 ##
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# for i in cat_col:
#     encoder = LabelEncoder()
#     x_train[i] = encoder.fit_transform(x_train[i])
#     x_test[i] = encoder.transform(x_test[i])

# print('인코딩후 ' ,x_train.shape,y_train.shape,x_test.shape)

x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)
print('인코딩후 ' ,x_train.shape,y_train.shape,x_test.shape)
x_test = x_test.reindex(columns=x_train.columns,fill_value=0)




# 진짜 분리
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score , mean_squared_error

# print(sklearn.metrics.__all__)

x_tr , x_val , y_tr , y_val = train_test_split(x_train,y_train,test_size=0.3,random_state=2023)
print('분리후 ' , x_tr.shape , x_val.shape , y_tr.shape , y_val.shape)

from lightgbm import LGBMRegressor
model = LGBMRegressor(random_state=2023) 
# model = RandomForestClassifier(random_state=2023)
model.fit(x_tr,y_tr)
pred = model.predict(x_val)

r2_sc = r2_score(pred,y_val)
rmse_sc = np.sqrt(mean_squared_error(pred,y_val))
print('스코어',r2_sc,rmse_sc)

result = model.predict(x_test)
# print(result)


# df = pd.DataFrame({'pred':result})
# df.to_csv('result.csv',index=False)

df2 = pd.DataFrame({'id':x_test_id ,'price':result})
df2.to_csv('submission.csv',index=False)
# print(pd.read_csv('submission.csv'))