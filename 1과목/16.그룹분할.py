# basic1 동일한 개수로 나이 순으로 3그룹으로 나눈 뒤 각 그룹의 중앙값을 더하시오

import pandas as pd
import numpy as np
df = pd.read_csv('basic1.csv')


df['range'] = pd.qcut(df['age'],q=3,labels = ['group1','group2','group3']) #3등분후 변수생성
group1 = df[df['range']=='group1']['age'].median()
group2 = df[df['range']=='group2']['age'].median()
group3 = df[df['range']=='group3']['age'].median()
print(group1 + group2 + group3)

