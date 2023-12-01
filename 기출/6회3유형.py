# 6회 3유형
import pandas as pd 
df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/adp/28/p7.csv')
df.head()

# age와 Cholesterol을 가지고 weight를 예측하는 선형 회귀 모델을 만들려고한다. age의 회귀 계수를 구하여라
from sklearn.linear_model import LinearRegression
# print(sklearn.linear_model.__all__)
x = df[['age','Cholesterol']]
y = df['weight']
model = LinearRegression()
model.fit(x,y)
print(model.coef_[0])

import statsmodels.api as sm
# print(sm.__all__)
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
print(model.summary())


# age가 고정일 때 Cholesterol와 weight가 선형관계에 있다는 가설을 유의수준 0.05하에 검정하라
from scipy import stats
result = stats.pearsonr(df['Cholesterol'],df['weight'])
print(result)

# age가 55 , Cholesterol 72.6 일때 예측 
age = 55
Cholesterol = 72.6
modelx = 74.8953 + (-0.0361*55) + (0.0819*72.6)
print(model.predict([1,55,72.6])) # 방법1
print(modelx) # 방법2

