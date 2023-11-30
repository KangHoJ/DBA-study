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
    
df = pd.read_csv("./adult.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='income', null_name='?')

# start
print('처음',X_train.shape, X_test.shape, y_train.shape)
# print(X_train.shape , X_test.shape , y_train.shape)
# print(X_train.head(5))
# print(X_train.info()) # 결측치는 없음 
# print(X_train.select_dtypes(exclude='object').columns)
col_category = ['workclass', 'education', 'marital.status', 'occupation','relationship', 'race', 'sex', 'native.country']
col_num = ['id', 'age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss','hours.per.week']

X_train['capital'] = X_train['capital.gain'] - X_train['capital.loss']
X_test['capital'] = X_test['capital.gain'] - X_test['capital.loss']
X_train = X_train.drop(['native.country','education','capital.gain','capital.loss'],axis=1)
X_test = X_test.drop(['native.country','education','capital.gain','capital.loss'],axis=1)

print('드랍한상태',X_train.shape, X_test.shape, y_train.shape)

X_test_id = X_test['id']

# X = pd.get_dummies(X_train)
# Test = pd.get_dummies(X_test)
# Y = (y_train['income']!='<=50K').astype(int)
# X , Test = X.align(Test,axis=1,join='outer',fill_value=0)
# print('인코딩완료상태',X.shape , Y.shape , Test.shape)

from sklearn.preprocessing import LabelEncoder
cols = ['workclass', 'marital.status', 'occupation','relationship', 'race', 'sex']
for i in cols:
	ll = LabelEncoder()
	X_train[i] = ll.fit_transform(X_train[i])
	X_test[i] = ll.transform(X_test[i])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_train)
Test = scaler.transform(X_test)
Y = (y_train['income']!='<=50K').astype(int)
print('인코딩완료상태',X.shape , Y.shape , Test.shape)

## 분리
# import sklearn.ensemble 
# print(sklearn.ensemble.__all__)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

x_tr , x_te , y_tr , y_te = train_test_split(X,Y,test_size=0.1,random_state=15)
print('분리 완료', x_tr.shape , x_te.shape , y_tr.shape , y_te.shape)

model = GradientBoostingClassifier()
model.fit(x_tr,y_tr)
pred = model.predict(x_te)
score = accuracy_score(y_te,pred)
print('정확도',score)
print(model.score(X,Y))


output = model.predict(Test)
print('결과',output)

submission= pd.DataFrame({'id':X_test_id,'Income':output})
pd.DataFrame({'id':X_test_id,'Income':output}).to_csv('adult.csv',index=True)
print(submission.head())
