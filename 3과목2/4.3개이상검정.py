import pandas as pd
from scipy import stats

# 세그룹 성적 평균이 같다고 할수있나? (아노바검정)
# 귀무가설 : 세 그룹 평균이 같다.

df=pd.DataFrame( {
				'A': [120, 135, 122, 124, 135, 122, 145, 160, 155, 142, 144, 135, 167],
				'B' : [110, 132, 123, 119, 123, 115, 140, 162, 142, 138, 135, 142, 160],
				'C' : [130, 120, 115, 122, 133, 144, 122, 120, 110, 134, 125, 122, 122]})

# 정규성 검정
result = stats.shapiro(df['A'])
print(result)
result = stats.shapiro(df['B'])
print(result)
result = stats.shapiro(df['C'])
print(result)

# 정규성 x 일떄 검정
result = stats.kruskal(df['A'],df['B'],df['C'])
print(result)

# 등분산 검정
result = stats.bartlett(df['A'],df['B'],df['C'])
print(result)

# 분산분석
result = stats.f_oneway(df['A'],df['B'],df['C'])
print(result)

#####################################################################################

# 만약 데이터가 다를 경우 

df2	=	pd.DataFrame( {
				'항목': ['A','A','A','A','A','A','A','A','A','A','A','A','A',
											'B','B','B','B','B','B','B','B','B','B','B','B','B',
											'C','C','C','C','C','C','C','C','C','C','C','C','C',],
				'value': [120, 135, 122, 124, 135, 122, 145, 160, 155, 142, 144, 135, 167,
													110, 132, 123, 119, 123, 115, 140, 162, 142, 138, 135, 142, 160,
													130, 120, 115, 122, 133, 144, 122, 120, 110, 134, 125, 122, 122]})

a = df2[df2['항목']=='A']['value']
b = df2[df2['항목']=='B']['value']
c = df2[df2['항목']=='C']['value']
result = stats.f_oneway(a,b,c)
print(result)