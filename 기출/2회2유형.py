import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

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
    
df = pd.read_csv("./Train.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='Reached.on.Time_Y.N', id_name='ID')

print(X_train.shape, X_test.shape, y_train.shape) # Reached.on.Time_Y.N 예측값 
##########################################################################################################
X_test_id = X_test['ID']

# 데이터 타입 분리
cat_col = X_train.select_dtypes(include='object').columns # ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
num_col = X_train.select_dtypes(exclude='object').columns 
# ['ID', 'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product','Prior_purchases', 'Discount_offered', 'Weight_in_gms']
print(cat_col) ,print(num_col)

# 결측치 확인
# print(X_train.isnull().sum()) # 결측치 x

# 이상치 확인
# print(X_train.describe()) # 크게 눈에띄는건 없음 

# 상관관계 확인
# print(X_train[num_col].corr())

# 상관관계를 확인하기 위해 합치기
# new = pd.merge(X_train,y_train,how='left',on='ID')
# new_nul_col = ['ID', 'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product','Prior_purchases', 'Discount_offered', 'Weight_in_gms','Reached.on.Time_Y.N'] 
# print(new[new_nul_col].corr()) # Customer_rating , Customer_care_calls , Prior_purchases ,Cost_of_the_Product 낮은영향

# 범주형 변수 보기
# print(X_train['Gender'].value_counts()) # 대체적으로 균일

X_train = X_train.drop(columns=['Customer_rating','Customer_care_calls','Cost_of_the_Product'],axis=1)
X_test = X_test.drop(columns=['Customer_rating','Customer_care_calls','Cost_of_the_Product'],axis=1)
print('drop후 shape' , X_train.shape, X_test.shape, y_train.shape) 

num_col = X_train.select_dtypes(exclude='object').columns
cat_col = X_train.select_dtypes(include='object').columns

# 인코딩 
# X_train = pd.get_dummies(X_train)
# X_test = pd.get_dummies(X_test)
# print('인코딩후 shape' , X_train.shape, X_test.shape, y_train.shape) 

#라벨 인코딩시 
for i in cat_col:
    ll = LabelEncoder()
    X_train[i] = ll.fit_transform(X_train[i])
    X_test[i] = ll.transform(X_test[i])

# 스케일링
sc = MinMaxScaler()
X_train[num_col] = sc.fit_transform(X_train[num_col])
X_test[num_col] = sc.transform(X_test[num_col])
print('스케일링후 shape' , X_train.shape, X_test.shape, y_train.shape) 


# 분리
x_tr , x_val , y_tr , y_val = train_test_split(X_train,y_train['Reached.on.Time_Y.N'],test_size=0.3,stratify=y_train['Reached.on.Time_Y.N'],random_state=2023)
print('분리후',x_tr.shape , x_val.shape , y_tr.shape , y_val.shape)

# roc_auc_score 평가
model = RandomForestClassifier(random_state=2023,n_estimators=75,max_depth=3)
model2 = LGBMClassifier()
model3 = XGBClassifier()

model.fit(x_tr,y_tr)
pred = model.predict_proba(x_val)
# print(pred[:,1])
score = roc_auc_score(y_val,pred[:,1])
print(score)

result = model.predict_proba(X_test)
# print(len(result))
# print(len(X_test_id))
df = pd.DataFrame({'id':X_test_id , 'Reached.on.Time_Y.N':result[:,1]})
df.to_csv('2_submission.csv',index=False)