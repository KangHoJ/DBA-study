import	pandas	as	pd
import	numpy	as	np
from	sklearn.datasets	import	load_diabetes
#	diabetes	데이터셋 로드
diabetes	=	load_diabetes()
x	=	pd.DataFrame(diabetes.data,	columns=diabetes.feature_names)
y	=	pd.DataFrame(diabetes.target)
y.columns	= ['target']
###############		실기환경

x = x[['age','sex','bmi']]
# print(y)

# import sklearn.linear_model 
from sklearn.linear_model import LinearRegression
# print(sklearn.linear_model.__all__)

model = LinearRegression()
model.fit(x,y)

print(model.score(x,y))
print(np.round(model.coef_[0][0],2))
print(np.round(model.coef_[0][1],2))
print(np.round(model.coef_[0][2],2))
print(np.round(model.intercept_,2))

# 최종 152.13348416 + 138.90391069674857age + -36.13526678248468sex + 926.9120121198846bmi

######################################################################################################
# 다른 모델 활용 방법

# from statsmodels.api as sm
import statsmodels.api as sm
print(dir(sm))

x= sm.add_constant(x)
model = sm.OLS(y,x).fit()
print(model.summary())
