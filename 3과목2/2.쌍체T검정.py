import pandas as pd
df	=pd.DataFrame( {
				'before': [120, 135, 122, 124, 135, 122, 145, 160, 155, 142, 144, 135, 167],
				'after' : [110, 132, 123, 119, 123, 115, 140, 162, 142, 138, 135, 142, 160] })

# 문제 : 혈압약을 먹기전과 후 차이가 있는지 쌍체 t 검정 실시
# 귀무가설 : 차이가 없다(같다)

from spicy import stats

# 정규성 검정 
result = stats.shapiro(df['after']-df['before']) # 정규성 만족
print(result)

# 만약 정규성 만족 x 일시
result = stats.wilcoxon(df['after']-df['before'],alternative='two-sided')
print(result)

# 쌍체 T 검정 
result = stats.ttest_rel(df['after'],df['before'],alternative='two-sided') # 차이가 있다
print(result) 

print('------------------------------------아래는 단일일경우')

# 문제 혈압약을 먹은 전 후 혈압이 감소했는지 확인
# 귀무가설 : 혈압약을 먹어도 혈압 감소 x (after-before) >=0

#정규성 검정
result = stats.shapiro(df['after']-df['before']) # 정규성 만족
print(result)

# 만약 정규성 만족 x 일시
result = stats.wilcoxon(df['after']-df['before'],alternative='less')
print(result)

# 쌍체 T 검정
result = stats.ttest_rel(df['after'],df['before'],alternative='less') # 감소했다고 할수있다 
print(result)


