# 1) 학생들의 키의 95% 신뢰구간을 구하고자 한다.
# 55명 학생들의 키에 대한 표본 평균을 구하여라(반올림하여 소숫점 3째자리까지
import pandas as pd 
import numpy as np
from scipy import stats

df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p3_1.csv')
# print(df)
print(round(df['height'].mean(),3))

# t분포 양쪽 꼬리에서의 t 값을 구하여라 (반올림하여 소수4째자리까지)
std = df['height'].std()
c_v = 0.95
ddof = len(df['height'])-1
t_value = stats.t.ppf((1+c_v)/2,ddof)  # (1+신뢰구간)/2 , 자유도
print(t_value)

# 95% 신뢰구간을 구하여라(print(lower,upper) 방식으로 출력, 각각의 값은 소숫점 이하 3째자리까지)
print(df['height'].mean() - t_value * std / np.sqrt(len(df['height']))) # 평균- t*시그마(std)/루트n
print(df['height'].mean() + t_value * std / np.sqrt(len(df['height']))) # 평균+ t*시그마(std)/루트n

##########################################################################################################

# 학과 평균 인원에 대한 값을 소숫점 이하 3자리까지 구하여라
import pandas as pd
from scipy import stats
df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p3_1_.csv')
# print(df.head())
print(round(df['학과'].value_counts().mean(),3))
# 다른 방법 

# 카이제곱검정 독립성 검정 통계량을 소숫점 이하 3자리까지 구하여라
table = pd.crosstab(df['학과'],df['성별'])
s,p,d,e = stats.chi2_contingency(table)
print(round(s,3))
print(round(p,3))

# 어느 학교에서 수학 시험을 본 학생 100명 중 60명이 60점 이상을 받았다.
# 이 학교의 수학 시험의 평균 점수가 50점 이상인지 95%의 신뢰 수준에서 검정하려한다.

import numpy as np
n = 100
p_hat = 0.6
p = 0.5
alpha = 0.05

# 검정 통계량 계산
z = (p_hat-p)/np.sqrt(p*(1-p)/n)
print(z)

# 유의수준 계산
p_value = round(1 - stats.norm.cdf(z),3)


######################################################################################################
'''
투약 후 체중에서 투약 전 체중을 뺏을 때 값은 일반 적으로 세가지 등급으로 나눈다. 
-3이하 : A등급, -3초과 0이하 : B등급, 0 초과 : C등급. 약 실험에서 A,B,C 그룹간의 인원 수 비율은 2:1:1로 알려져 있다.
 위 데이터 표본은 각 범주의 비율에 적합한지 카이제곱 검정하려한다.
'''
import pandas as pd
from scipy import stats
df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p3_1.csv')

# A등급에 해당하는 유저는 몇명인지 확인하라
df['new'] = df['투약후'] - df['투약전']
def medical(x):
    if x < -3:
        return 'A등급'
    elif  -3  < x < 0 :
        return 'B등급'
    else:
        return 'C등급'
df['new'] = df['new'].map(lambda x : medical(x))
print(df['new'].value_counts()) # 121
# df['new'] = df['new'].apply(medical)

# 카이제곱검정 통계량을 반올림하여 소숫점 이하 3째자리까지 구하여라
f_obs = [121,67,47]
f_exp = [235*0.5,235*0.25,235*0.25]
t, p  = stats.chisquare(f_obs=f_obs,f_exp=f_exp)
print(round(t,3))
print(round(p,3))


# 생산한 기계들의 rpm 값들을 기록한 데이터이다. 대응 표본 t 검정을 통해 B공장 제품들이 A 공장 제품들보다 rpm이 높다고 말할 수 있는지 검정하려한다
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p3_2_.csv')

# A,B 공장 각각 정규성을 가지는지 샤피로 검정을 통해 확인하라. (각 공장의 pvalue 출력할 것)
a = df[df['group']=='A']['rpm']
b = df[df['group']=='B']['rpm']
s,p = stats.shapiro(a)
s2,p2 = stats.shapiro(b)
print(p , p2 )

# A,B 공장 생산 제품의 rpm은 각각 정규성을 가지는지 샤피로 검정을 통해 확인하라. (각 공장의 pvalue 출력할 것)
s , p =stats.levene(a,b)
print(p)

# 대응 표본=쌍체 t 검정을 통해 B공장 제품들의 rpm이 A 공장 제품의 rpm보다 크다고 말할 수 있는지 검정하라. 
# pvalue를 소숫점 이하 3자리까지 출력하고 귀무가설, 대립가설 중 하나를 출력하라*
# 대립가설 : B>A
t , p = stats.ttest_rel(a,b,alternative='less')
print(round(p,3))

import pandas as pd
from scipy import stats
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e2_p3_1.csv')
print(df.head())

# 122마리의 height 평균값을 m(미터) 단위로 소숫점 이하 5자리까지 실수 값만 출력하라
df['height'] = df['height'].str.replace('cm',' ').astype('float')
print(round(df['height'].mean(),3))

# 모집단의 평균 길이가 30cm 인지 확인하려 일표본 t 검정을 시행하여 확인하려한다. 검정통계량을 소숫점 이하 3째자리까지 구하여라
t , p = stats.ttest_1samp(df['height'],30)
print(round(t,3))
print(round(p,3))


# 조사결과 70%의 성인 남성이 3년 동안에 적어도 1번 치과를 찾는다고 할때, 21명의 성인 남성이 임의로 추출되었다고 하자.
# 21명 중 16명 미만이 치과를 찾았을 확률(반올림하여 소숫점 이하 3자리)
# p(x<k)계산
n = 21
p = 0.7
k = 16
prob = stats.binom.cdf(k-1,n,p)
print(round(prob,3))

n=21
p=0.7
k=19
# 적어도 19명이 치과를 찾았을 확률(반올림하여 소숫점 이하 3자리)
# 1-p(x<k)계산
prob2 = 1-stats.binom.cdf(k-1,n,p)
print(round(prob2,3))


import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')

# 가장 많은 광고를 집행했던 날짜는 언제인가? (데이터 그대로 일(숫자),달(영문)으로 표기) 
print(df.groupby('month')['campaign'].sum().idxmax())
df = df[df['month']=='may']
print(df.groupby('day')['campaign'].sum().idxmax())

# 데이터의 job이 unknown 상태인 고객들의 age 컬럼 값의 정규성을 검정하고자 한다. 
# 샤피로 검정의 p-value값을 구하여라
from scipy import stats
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')
df = df[df['job']=='unknown']
t , p = stats.shapiro(df['age'])
print(p)

## age와 balance의 상관계수를 구하여라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')
print(df[['age','balance']].corr())

# y 변수와 education 변수는 독립인지 카이제곱검정을 통해 확인하려한다. p-value값을 출력하라
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')
table = pd.crosstab(df['y'],df['education'])
s , p , ddof , exp = stats.chi2_contingency(table)
print(p)

