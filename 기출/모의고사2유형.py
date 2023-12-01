import pandas as pd
X_test = pd.read_csv("./X_test_1.csv")
X_train = pd.read_csv("./X_train_1.csv")
y_train = pd.read_csv("./y_train_1.csv")
print(X_train.shape,X_test.shape,y_train.shape)

# print(X_train.head(),y_train.head())
train = pd.merge(X_train,y_train,how='left',on='enrollee_id')
print(train.shape,X_test.shape)


####### EDA ########### 
# 범주형 수치형 분리
cat_col = train.select_dtypes(include='object').columns 
# ['city', 'gender', 'relevent_experience', 'enrolled_university','education_level', 'major_discipline', 'experience', 'company_size','company_type', 'last_new_job']
num_col = train.select_dtypes(exclude='object').columns
# ['enrollee_id', 'city_development_index', 'training_hours'] + target


# 결측치 확인
# print(train.isnull().sum())
# print(train['gender'].value_counts())

# gender 결측값 채우기
# print(train.isnull().sum(),X_test.isnull().sum())
n_g_tr = train['gender'].isnull().sum()
n_g_te = X_test['gender'].isnull().sum()
train['gender'] = train['gender'].fillna('Male',limit=int(n_g_tr*0.7))
train['gender'] = train['gender'].fillna('Female',limit=int(n_g_tr*0.2))
train['gender'] = train['gender'].fillna('Other')
X_test['gender'] = X_test['gender'].fillna('Male',limit=int(n_g_te*0.7))
X_test['gender'] = X_test['gender'].fillna('Female',limit=int(n_g_te*0.2))
X_test['gender'] = X_test['gender'].fillna('Other')


# education_level , enrolled_university 결측값행날리기
print(train.isnull().sum(),X_test.isnull().sum())
# print(train['education_level'].value_counts())


train['education_level'] = train['education_level'].fillna('etc')
X_test['education_level'] = X_test['education_level'].fillna('etc')
train['enrolled_university'] = train['enrolled_university'].fillna('etc')
X_test['enrolled_university'] = X_test['enrolled_university'].fillna('etc')
train['experience'] = train['experience'].fillna(train['experience'].mode()[0])
X_test['experience'] = X_test['experience'].fillna(train['experience'].mode()[0])
train['last_new_job'] = train['last_new_job'].fillna(train['last_new_job'].mode()[0])
X_test['last_new_job'] = X_test['last_new_job'].fillna(train['last_new_job'].mode()[0])
# print(train.isnull().sum())
print('결측값 채우기 후 ' , train.shape,X_test.shape)
print(train.isnull().sum(),X_test.isnull().sum())

train = train.drop(columns=['major_discipline','company_size','company_type'])
X_test = X_test.drop(columns=['major_discipline','company_size','company_type'])
print('드랍 후 ' , train.shape,X_test.shape)
print(train.isnull().sum(),X_test.isnull().sum())


train = train[train['training_hours']<300]
print(' 전처리 후 ' , train.shape,X_test.shape)

# # 상관관계
# # print(train[num_col].corr()) # training_hours , enrollee_id 상관관계 낮았음

# # 분리
X_train = train.drop('target',axis=1)
Y_train = train['target']
X_test = X_test
print('분리 후 :', X_train.shape,X_test.shape,Y_train.shape)


# #인코딩
# X_train = pd.get_dummies(X_train)
# X_test = pd.get_dummies(X_test)
# print('인코딩 후 :', X_train.shape,X_test.shape,Y_train.shape)

# X_test = X_test.reindex(columns=X_train.columns,fill_value=0)
# print('교정 후 :', X_train.shape,X_test.shape,Y_train.shape)


cat_col = X_train.select_dtypes(include='object').columns
from sklearn.preprocessing import LabelEncoder
for i in cat_col:
    ll = LabelEncoder()
    X_train[i] = ll.fit_transform(X_train[i])
    X_test[i] = ll.transform(X_test[i])
print('인코딩 후 :', X_train.shape,X_test.shape,Y_train.shape)


# # 분리 , 모델 , 평가
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import 
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

x_tr , x_val , y_tr , y_val = train_test_split(X_train,Y_train,test_size=0.3,stratify=Y_train,random_state=2023)
print(x_tr.shape , x_val.shape , y_tr.shape , y_val.shape)


model = XGBClassifier(random_state=2023,max_depth=5,n_estimators=100)
model.fit(X_train,Y_train)
tr_pred = model.predict_proba(X_train)
te_pred= model.predict_proba(x_val)
score = roc_auc_score(Y_train,tr_pred[:,1])
score2 = roc_auc_score(y_val,te_pred[:,1])
print(score , score2)

predi = model.predict_proba(X_test)
pred = predi[:,1]
# pd.DataFrame({'id': X_test.enrollee_id, 'target': pred}).to_csv('003000000.csv', index=False)
df = pd.DataFrame({'enrollee_id': X_test['enrollee_id'],'target':pred})
print(df)

# import pickle
# with open("./answer.pickle", "rb") as file:
#     ans = pickle.load(file)
#     ans = pd.DataFrame(ans)
# print(ans)
# print(roc_auc_score(ans['target'], pred))

# 컬럼이름 바꾸고 싶을때 
# df.rename()