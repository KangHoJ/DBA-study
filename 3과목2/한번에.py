import pandas as pd
import	numpy as np
from sklearn.datasets import load_diabetes
import seaborn as sns

# 문제 mpg열 평균이 20과 같다고 할수있는지 검정 
# 귀무가설 : 20과 같다 
df = pd.read_csv('./mtcars.csv')
from scipy import stats
result_nor = stats.shapiro(df['mpg'])
result_norx = stats.wilcoxon(df['mpg']-20,alternative='two-sided')
print('비정규 단일 ',result_norx)
result = stats.ttest_1samp(df['mpg'],20,alternative='two-sided')
print('단일표본',result)

print('------------------'*5)

# 문제 : 혈압약을 먹기전과 후 차이가 있는지 쌍체 t 검정 실시
df	=pd.DataFrame( {
				'before': [120, 135, 122, 124, 135, 122, 145, 160, 155, 142, 144, 135, 167],
				'after' : [110, 132, 123, 119, 123, 115, 140, 162, 142, 138, 135, 142, 160] })

from scipy import stats
result_nor1 = stats.shapiro(df['after'] - df['before'])
result_norx = stats.wilcoxon(df['after'] - df['before'],alternative='two-sided')
print('비정규 쌍체t',result_norx)
result = stats.ttest_rel(df['after'],df['before'],alternative='two-sided')
print('쌍체t',result)

print('------------------'*5)

# 문제 혈압약을 먹은 전 후 혈압이 감소했는지 확인
# 귀무가설 : 혈압약을 먹어도 혈압 감소 x
from spicy import stats
result_nor1 = stats.shapiro(df['after']-df['before'])
result_norx = stats.wilcoxon(df['after']-df['before'],alternative='less')
print('비정규 쌍체t',result_norx)
result = stats.ttest_rel(df['after'],df['before'],alternative='less')
print('쌍체t',result)

print('------------------'*5)

# 두 그룹의 혈압 평균이 다르다고 할수있는지 검정
# 귀무가설 : 두 그룹의 평균은 같다 
df	=	pd.DataFrame( {
				'A': [120, 135, 122, 124, 135, 122, 145, 160, 155, 142, 144, 135, 167],
				'B' : [110, 132, 123, 119, 123, 115, 140, 162, 142, 138, 135, 142, 160] })

from scipy import stats
nor1 = stats.shapiro(df['A'])
nor2 = stats.shapiro(df['B'])
nox = stats.ranksums(df['A'],df['B'],alternative='two-sided')
print('독립 정규성 x 일때',nox)
var = stats.bartlett(df['A'],df['B'])
result = stats.ttest_ind(df['A'],df['B'],equal_var=True,alternative='two-sided')
print('독립',result)
# print(stats.__all__)


print('------------------'*5)

# A 그룹의 혈압평균이 B그룹보다 크다고 할 수 있나 ?
# 귀무가설 A<=B
nor1 = stats.shapiro(df['A'])
nor2 = stats.shapiro(df['B'])
norx = stats.ranksums(df['A'],df['B'],alternative='greater')
print('독립 정규성 x 일때',nox)
var = stats.bartlett(df['A'],df['B'])
result = stats.ttest_ind(df['A'],df['B'],equal_var=True,alternative='greater')
print('독립',result)

print('------------------'*5)
# 세그룹 성적 평균이 같다고 할수있나? 
# 귀무가설 : 세 그룹 평균이 같다.
df=pd.DataFrame( {
				'A': [120, 135, 122, 124, 135, 122, 145, 160, 155, 142, 144, 135, 167],
				'B' : [110, 132, 123, 119, 123, 115, 140, 162, 142, 138, 135, 142, 160],
				'C' : [130, 120, 115, 122, 133, 144, 122, 120, 110, 134, 125, 122, 122]})

nor1 = stats.shapiro(df['A'])
nor2 = stats.shapiro(df['B'])
nor3 = stats.shapiro(df['C'])
norx = stats.kruskal(df['A'],df['B'],df['C'])
print('3개이상 정규성 x 일때',norx)
var = stats.bartlett(df['A'],df['B'],df['C'])
result = stats.f_oneway(df['A'],df['B'],df['C'])
print('3개이상 정규성 o 일때',result)

print('------------------'*5)
# 만약 데이터가 다를 경우 
df2	=pd.DataFrame( {
				'항목': ['A','A','A','A','A','A','A','A','A','A','A','A','A',
											'B','B','B','B','B','B','B','B','B','B','B','B','B',
											'C','C','C','C','C','C','C','C','C','C','C','C','C',],
				'value': [120, 135, 122, 124, 135, 122, 145, 160, 155, 142, 144, 135, 167,
													110, 132, 123, 119, 123, 115, 140, 162, 142, 138, 135, 142, 160,
													130, 120, 115, 122, 133, 144, 122, 120, 110, 134, 125, 122, 122]})


a = df2[df2['항목']=='A']['value']
b = df2[df2['항목']=='B']['value']
c = df2[df2['항목']=='C']['value']
var = stats.bartlett(a,b,c)
nox = stats.kruskal(a,b,c)
print('정규성x일때 3개',nox)
result = stats.f_oneway(a,b,c)
print('정규성일때 3개',result)


print('------------------'*5)
# 연령대에 따라 먹는 아이스크림의 차이가 있는지 독립성 검정을 실시하시오
# 귀무가설 : 독립이다 / 대립가설 : 관련이있다
df	=	pd.DataFrame({
				'아이스크림' : ['딸기','초코','바닐라','딸기','초코','바닐라'],
				'연령' : ['10대','10대','10대','20대','20대','20대'],
				'인원' : [200,190,250,220,250,300]
				})

from scipy import stats
# print(stats.__all__)
table = pd.crosstab(index=df['연령'],columns=df['아이스크림'],values=df['인원'],aggfunc=sum)
result = stats.chi2_contingency(table)
print(result)

print('------------------'*5)
# 타이타닉에서 sex,survived 독립성 검정
import seaborn as sns
df = sns.load_dataset('titanic')

table = pd.crosstab(df['sex'],df['survived'])
result = stats.chi2_contingency(table)
print(result)


print('------------------'*5)
# 'age','sex','bmi' 다중 회귀
diabetes	=	load_diabetes()
x	=	pd.DataFrame(diabetes.data,	columns=diabetes.feature_names)
y	=	pd.DataFrame(diabetes.target)
y.columns	= ['target']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
x = x[['age','sex','bmi']]
y
model.fit(x,y)
print('계수',model.coef_)
print('y절편',model.intercept_)


# 방법2 
import statsmodels.api as sm
x = sm.add_constant(x)
summaryO = sm.OLS(y,x).fit()
# print(summaryO.summary())

print('------------------'*5)
# 'survived','sex','sibsp','fare' 로지스틱 회귀
df = sns.load_dataset('titanic')
x = df[['sex','sibsp','fare']]
y = df['survived']
x['sex'] = x['sex'].map({'female':0,'male':1})

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='none')
model.fit(x,y)
print('계수',model.coef_)
print('y절편',model.intercept_)

print('------------------'*5)
# # 문제 3 : sibsp변수가 한단위 증가할때마다 생존할 오즈가 몇배 증가하는지 구하시오
print('odds증가비',np.exp(model.coef_[0][1]))


# 방법2
df = sns.load_dataset('titanic')
df = df[['survived','sex','sibsp','fare']]
df['sex'] = df['sex'].map({'female':0,'male':1})
x = df.drop(['survived'],axis=1)
y = df['survived']


import statsmodels.api as sm
x = sm.add_constant(x)
model = sm.Logit(y,x).fit()
# print(model.summary())

print('------------------'*5)
# bmi , target 선형관계 검정
diabetes	=	load_diabetes()
x	=	pd.DataFrame(diabetes.data,	columns=diabetes.feature_names)
y	=	pd.DataFrame(diabetes.target)
y.columns	= ['target']

result = stats.pearsonr(x['bmi'],y['target'])
print('선형검정',result)

print('------------------'*5)
# 문제 : 랜덤 박스에 상품	A,B,C,D가 들어있다. 다음은 랜덤박스에서	100번 상품을 꺼냈을 때의 상품 데이터라고 
# 할 때 상품이 동일한 비율로 들어있다고 할 수 있는지 검정
# 귀무가설 : 동일한 비율로 들어있다
row1	= [30, 20, 15, 35]
df=pd.DataFrame([row1],columns=['A','B','C','D'])

f_obs = [30,20,15,35]
f_exp = [25,25,25,25]
result = stats.chisquare(f_obs=f_obs,f_exp=f_exp)
print('카이제곱',result)

print('------------------'*5)
# 문제 : A 30%,	B 15%,	C 55%	비율로 들어있다고 할 수 있는지 검정해보시오
row1	= [50,25,75]
df	=	pd.DataFrame([row1],columns=['A','B','C'])

f_obs = [50,25,75]
f_exp = [150*0.3,150*0.15,150*0.55]
result = stats.chisquare(f_obs=f_obs,f_exp=f_exp)
print('카이제곱2',result)