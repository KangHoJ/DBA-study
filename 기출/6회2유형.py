import pandas as pd
import numpy as np
pd.options.display.max_columns=1000
train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/ep6_p2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/ep6_p2_test.csv')
# print(train['General_Health'].value_counts())

print('처음',train.shape , test.shape)

# EDA 시작 #########

# 결측치 확인
# print(train.isnull().sum(),test.isnull().sum())

# 칼럼 나누기
cat_col = train.select_dtypes(include='object').columns 
# ['ID', 'General_Health', 'Checkup', 'Exercise', 'Heart_Disease','Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis','Sex', 'Age_Category', 'Smoking_History']
num_col = train.select_dtypes(exclude='object').columns
# ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption','FriedPotato_Consumption']
# print(cat_col)
# print(num_col)

# 이상치 확인
# print(train.describe())
# print(train[train['Weight_(kg)']>200]) # 10 명
# print(train[train['BMI']>70]) # 12명
train['BMI'] = np.where(train['BMI']>70,30,train['BMI'])
test['BMI'] = np.where(test['BMI']>70,30,test['BMI'])
train['healty'] = train['Fruit_Consumption'] + train['Green_Vegetables_Consumption'] + train['FriedPotato_Consumption']
test['healty'] = test['Fruit_Consumption'] + test['Green_Vegetables_Consumption'] + test['FriedPotato_Consumption']

# print(train[train['Weight_(kg)']>200]) 

print('이상치 처리후 ' , train.shape , test.shape)

# 상관관계 
# print(train[num_col].corr()) # bmi , weight 다중 공선성 의심
train = train.drop(columns=['Weight_(kg)','Fruit_Consumption','ID'],axis=1)
test = test.drop(columns=['Weight_(kg)','Fruit_Consumption','ID'],axis=1)

print('드랍후 ' , train.shape , test.shape)

# cat_col확인
# for i in cat_col:
#     print(f'{i}고유값',train[i].unique())
#     print(f'{i}길이',len(train[i].unique()))

X_train = train.drop('General_Health',axis=1)
y_train = train['General_Health']
X_test = test

num_col = X_train.select_dtypes(exclude='object').columns
cat_col = X_train.select_dtypes(include='object').columns

# 스케일링 StandardScaler / MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
# for i in num_col:
#     sc = StandardScaler()
#     X_train[i] = sc.fit_transform(X_train[[i]])
#     X_test[i] = sc.transform(X_test[[i]])

for i in cat_col:
    ll = LabelEncoder()
    X_train[i] = ll.fit_transform(X_train[i])
    X_test[i] = ll.transform(X_test[i])

# 인코딩
# X_train = pd.get_dummies(X_train)
# X_test = pd.get_dummies(X_test)
# X_test = X_test.reindex(columns=X_train.columns,fill_value=0)
print('인코딩후',X_train.shape,X_test.shape,y_train.shape)
# print(X_train.columns)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


x_tr , x_te , y_tr , y_te = train_test_split(X_train,y_train,test_size=0.3,stratify=y_train,random_state=2023)
print('분리후',x_tr.shape , x_te.shape , y_tr.shape , y_te.shape)

model = GradientBoostingClassifier(random_state=2023,n_estimators=100)
model.fit(X_train,y_train)
pred_tr = model.predict(x_tr)
pred_te = model.predict(x_te)

score = f1_score(pred_tr,y_tr,average='weighted')
score2 = f1_score(pred_te,y_te,average='weighted')
print(score,score2)