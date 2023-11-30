import pandas as pd 
df	=	pd.DataFrame( {
				'A': [120, 135, 122, 124, 135, 122, 145, 160, 155, 142, 144, 135, 167],
				'B' : [110, 132, 123, 119, 123, 115, 140, 162, 142, 138, 135, 142, 160] })

# 두 그룹의 혈압 평균이 다르다고 할수있는지 검정
# 귀무가설 : 두 그룹의 평균은 같다 
from spicy import stats

# 정규성 검정 
result1 = stats.shapiro(df['A'])
result2 = stats.shapiro(df['B'])
print('정규성 검정 : ' , result1 , result2) 

# 정규성 x 일때
result3 = stats.ranksums(df['A'],df['B'],alternative='two-sided')
print('정규성x 일때',result3)

# 등분산성 검정 
result4 = stats.bartlett(df['A'],df['B'])
print('등분산 검정', result4)

result = stats.ttest_ind(df['A'],df['B'],equal_var=True,alternative='two-sided') # 만약 등분산 x 경우 equal_var =False
print('독립표본 테스트',result)

print('------------------------------------아래는 단일일경우')

# A 그룹의 혈압평균이 B그룹보다 크다고 할 수 있나 ?
# 귀무가설 A<=B

result5 = stats.shapiro(df['A']) # 정규성검정
result6 = stats.shapiro(df['B']) # 정규성검정
result7 = stats.ranksums(df['A'],df['B'],alternative='greater') #정규성x이면 검정방법
result8 = stats.bartlett(df['A'],df['B']) # 등분산 검정
result9 = stats.ttest_ind(df['A'],df['B'],equal_var=True,alternative='greater') # 독립표본 검정 
print(result5,result6,result7,result8)
print(result9)