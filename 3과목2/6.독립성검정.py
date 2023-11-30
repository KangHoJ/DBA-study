import pandas as pd
import numpy as np

# 연령대에 따라 먹는 아이스크림의 차이가 있는지 독립성 검정을 실시하시오
# 귀무가설 : 독립이다 / 대립가설 : 관련이있다

row1,row2 = [200, 190, 250], [220, 250, 300]
df = pd.DataFrame([row1,row2],	columns=['딸기','초코','바닐라'],index=['10대', '20대'])
print(df)

from scipy import stats

result = stats.chi2_contingency(df)
print(result)


###############################################################################################

# 만약 데이터가 다르게 주어질때(아이스크림,연령,인원이 따로 합쳐지지않고)

df	=	pd.DataFrame({
				'아이스크림' : ['딸기','초코','바닐라','딸기','초코','바닐라'],
				'연령' : ['10대','10대','10대','20대','20대','20대'],
				'인원' : [200,190,250,220,250,300]
				})
df

table = pd.crosstab(index=df['연령'],columns=df['아이스크림'],values=df['인원'],aggfunc=sum)
print(table)

result = stats.chi2_contingency(table)
print(result)

#######################################################################################
df	=	pd.DataFrame({'아이스크림':['딸기','초코','바닐라','딸기','초코','바닐라'],
                        '연령':['10대','10대','10대','20대','20대','20대']})

table = pd.crosstab(df['연령'],df['아이스크림'])
result = stats.chi2_contingency(table)
########################################################################################

# 타이타닉에서 sex,survived 독립성 검정
import seaborn as sns
df = sns.load_dataset('titanic')

table = pd.crosstab(df['sex'],df['survived'])
result = stats.chi2_contingency(table)
print('titanic',result)


