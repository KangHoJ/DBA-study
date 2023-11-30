import pandas as pd
import numpy as np
import seaborn as sns
df = sns.load_dataset('titanic')

df = df[['survived','sex','sibsp','fare']]

df['sex'] = df['sex'].map({'female':1,'male':0})
print(df['sex'])

x = df.drop(['survived'],axis=1)
y = df['survived']

# import sklearn.linear_model 
from sklearn.linear_model import LogisticRegression
# print(sklearn.linear_model.__all__)

model = LogisticRegression(penalty=None) # 꼭 penalty=None을 해야함
model.fit(x,y)
print(np.round(model.coef_,3))
print(model.intercept_)

# 문제 3 : sibsp변수가 한단위 증가할때마다 생존할 오즈가 몇배 증가하는지 구하시오
print(round(np.exp(model.coef_[0,1]),3))


import statsmodels.api as sm
x = sm.add_constant(x)
model  = sm.Logit(y,x).fit()
summary = model.summary()
print(summary)