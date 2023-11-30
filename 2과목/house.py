import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name
    
    if null_name != "":
        df[df == null_name] = np.nan
    
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=2021)
    
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[target])

    
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[target])
    return X_train, X_test, y_train, y_test 
    
df = pd.read_csv("./house.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='SalePrice', id_name='Id')

print('시작 shape',X_train.shape, X_test.shape, y_train.shape)
print(X_test.head())

# print(X_train.isnull().sum().sort_values(ascending=False)[:10])
X_train = X_train.select_dtypes(exclude='object')
X_test = X_test.select_dtypes(exclude='object')


print('수치형만:',X_train.shape , X_test.shape)
# print(X_train.isnull().sum())
# print(y_train.head())

X_test_id = X_test['Id'] # 나중에 데이터프레임 형성을 위해 빼놓기

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# import sklearn.impute
# from sklearn.impute import S
# print(sklearn.impute.__all__)
print('imputer 적용 후 ' , X_train.shape , X_test.shape , y_train.shape)


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score , mean_squared_error

Sclaer = MinMaxScaler()
X = Sclaer.fit_transform(X_train)
Test = Sclaer.transform(X_test)
Y=y_train['SalePrice']

print('sclaer 적용 후 ', X.shape , Test.shape , Y.shape )

x_tr , x_val , y_tr , y_val = train_test_split(X,Y,test_size=0.1,random_state=15)
print(x_tr.shape , x_val.shape , y_tr.shape , y_val.shape)

model = XGBRegressor()
model.fit(x_tr,y_tr)
pred = model.predict(x_val)

# 평가
score = r2_score(y_val,pred)
score2 = mean_squared_error(y_val,pred)
print('r2_score',score)
print('mean_squared_error' , np.sqrt(score2))

output = model.predict(X_test)
# print(output)
df = pd.DataFrame({'Id': X_test_id, 'SalePrice': output})
print(df.head())


