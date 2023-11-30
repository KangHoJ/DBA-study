import pandas as pd
import numpy as np
df = pd.read_csv('basic1.csv')
pd.options.display.max_columns=None

'''
데이터셋(basic1.csv)의 'f5' 컬럼을 기준으로 상위 10개의 데이터를 구하고,
'f5'컬럼 10개 중 최소값으로 데이터를 대체한 후,
'age'컬럼에서 80 이상인 데이터의'f5 컬럼 평균값 구하기
'''
m = df['f5'].sort_values(ascending=False).head(10).min()
print(df['f5'].sort_values(ascending=False).head(10).min())
df['f5']  = np.where(df['f5']>=m , m , df['f5'])
print(df[df['age']>=80]['f5'].mean())

# 방법2 
# for i in [10, 97, 9, 76, 98, 91, 86, 71, 11, 19]:
#     df['f5'].iloc[i] = m

# print(df['f5'].sort_values(ascending=False).head(10))
# print(df[df['age']>=80]['f5'].mean())

# 문제 2 
'''
데이터셋(basic1.csv)의 앞에서 순서대로 70% 데이터만 활용해서,
'f1'컬럼 결측치를 중앙값으로 채우기 전후의 표준편차를 구하고
두 표준편차 차이 계산하기
'''
df = pd.read_csv('basic1.csv')

df_70 = len(df)*0.7
df = df.iloc[:int(df_70)]
b_median = df['f1'].std()
df['f1'] = df['f1'].fillna(df['f1'].median())
a_median = df['f1'].std()
print(b_median - a_median)

# 문제 3 
'''
dsd
데이터셋(basic1.csv)의 'age'컬럼의 이상치를 더하시오! 단, 평균으로부터 '표준편차*1.5'를 벗어나는 영역을 이상치라고 판단함
'''
df = pd.read_csv('basic1.csv')
age_std = df['age'].std()
age_mean = df['age'].mean()
outline1 = age_mean+1.5*age_std
outline2 = age_mean-1.5*age_std

print(age_std,age_mean,outline1)
print(age_std,age_mean,outline2)

df_out = df[(df['age']>outline1) | (df['age']<outline2)]
print(df_out['age'].sum())
