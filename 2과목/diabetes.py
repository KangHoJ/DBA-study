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
    
df = pd.read_csv("./diabetes.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='Outcome')
print(X_test.head())








# 데이터 확인 (start)
print(X_train.shape, X_test.shape, y_train.shape)
# print(X_train.head())
# print(X_train.info())
# print(X_train.columns)
col = ['id', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Q1 = X_train['Glucose'].quantile(0.25)
# Q3 = X_train['Glucose'].quantile(0.75)
# IQR = Q3-Q1
# outliner= X_train[(X_train['Glucose']<(Q1-1.5*IQR)) | (X_train['Glucose']>(Q3+1.5*IQR))]
# print(outliner.index)

# print(X_train.corr()) # 다중 공선성 의심은 딱히 없다 

X_train = X_train.drop(['DiabetesPedigreeFunction'],axis=1)
X_test = X_test.drop(['DiabetesPedigreeFunction'],axis=1)
# print('drop후 shape', X_train.shape, X_test.shape, y_train.shape)

# import sklearn.preprocessing
# print(sklearn.preprocessing.__all__)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
for i in [StandardScaler,MinMaxScaler]:
    sc = i()
    X = sc.fit_transform(X_train)
    Y = y_train['Outcome']
    Test = sc.fit_transform(X_test)
    print(f'{i}sclaer까지 완료', X.shape, Test.shape, y_train.shape)

    # 모델 평가를 위한 분리 
    from sklearn.model_selection import train_test_split
    x_tr , x_val , y_tr , y_val = train_test_split(X,Y,test_size=0.3,random_state=15)
    print(x_tr.shape , x_val.shape , y_tr.shape , y_val.shape)

    # 모델 학습 및 예측 
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    for j in [GradientBoostingClassifier,DecisionTreeClassifier]:
        model = j(random_state=10,max_depth=4)
        model.fit(x_tr,y_tr)
        pred = model.predict(x_val)

        # 모델 평가
        from sklearn.metrics import accuracy_score
        print(f'{j}모델 평가 정확도',accuracy_score(pred,y_val))

        # import sklearn.ensemble
        # print(sklearn.ensemble.__all__)

        final_pred = model.predict(Test)
        print(f'{j}모델 정확도',model.score(X,Y))
        # pd.DataFrame({'id': X_test.id, 'gender': final_pred}).to_csv('003000000.csv', index=False)