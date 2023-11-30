'''
basic1 데이터와 basic3 데이터를 'f4'값을 기준으로 병합하고,
병합한 데이터에서 r2결측치를 제거한다음, 앞에서 부터 20개 데이터를 선택하고 'f2'컬럼 합을 구하시오
'''

import pandas as pd
import numpy as np
b1 = pd.read_csv('basic1.csv')
b2 = pd.read_csv('basic3.csv')

df = b1.merge(b2,how='left',on='f4')
df = df.dropna(subset=['r2']).head(20)
print(df['f2'].sum())


df_merge = pd.merge(left=b1,right=b2,how='left',on='f4')
df_merge = df_merge.dropna(subset='r2').head(20)
print(df['f2'].sum())