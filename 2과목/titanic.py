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
    



df = pd.read_csv("./train.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='Survived', id_name='PassengerId')

print(X_train.describe())
pd.options.display.max_columns=None
print(X_train.describe())


print('인코딩전 :', X_train.shape, X_test.shape, y_train.shape)
print(X_test.head())


X_train = X_train.drop(['Ticket', 'Name','Cabin','Age'], axis=1)
X_test = X_test.drop(['Ticket', 'Name','Cabin','Age'], axis=1)
print('drop후:',X_train.shape , X_test.shape)
print(X_train.isnull().sum())

Y = y_train['Survived'] 
X = pd.get_dummies(X_train)
test = pd.get_dummies(X_test)
print('인코딩 후:',X.shape , Y.shape , test.shape)

# 만약 교정이 필요할시
# X, test = X.align(test, axis=1, join='outer', fill_value=0)
# test = test.reindex(columns=X_train.columns , fill_value=0)

# print('교정 후:',X.shape , Y.shape , test.shape)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
X_tr , X_val , y_tr , y_val = train_test_split(X.values,Y.values,test_size=0.3,stratify=Y,random_state=25)
print(X_tr.shape, X_val.shape , y_tr.shape , y_val.shape)

# model = LGBMClassifier()
model = RandomForestClassifier()
model.fit(X_tr,y_tr)
pred = model.predict(X_val)
print('스코어',accuracy_score(pred,y_val))

pred = model.predict(test)
# print(pred)

output = pd.DataFrame({'PassengerId': X_test['PassengerId'], 'Survived': pred})
print(output.head(5))
output.to_csv('1234567.csv', index=False)

# from sklearn.model_selection import GridSearchCV
# params  = {'max_depth':[6,8,10],
#            'n_estimators':[30,70,100],
#             'min_samples_leaf':[1,2,3]}
# grid_model = GridSearchCV(model,param_grid=params,cv=10)
# grid_model.fit(X_tr,y_tr)
# print(grid_model.best_params_)
# print(grid_model.best_score_)