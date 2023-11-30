'''
수치형 변수 변환하기
주어진 데이터에서 20세 이상인 데이터를 추출하고 'f1'컬럼을 결측치를 최빈값으로 채운 후, f1 컬럼의 여-존슨과 박스콕스 변환 값을 구하고, 
두 값의 차이를 절대값으로 구한다음 모두 더해 소수점 둘째 자리까지 출력(반올림)하시오
'''

import pandas as pd
import numpy as np
df = pd.read_csv('basic1.csv')

df = df[df['age']>=20]
df['f1'] = df['f1'].fillna(df['f1'].mode()[0])
# print(df['f1'].mode()[0])
from sklearn.preprocessing import power_transform
pt = power_transform()

df['y'] = power_transform(df[['f1']],standardize=False)
df['b'] = power_transform(df[['f1']], method='box-cox')
