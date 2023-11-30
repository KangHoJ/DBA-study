'''
min-max스케일링 기준 상하위 5% 구하기
주어진 데이터에서 'f5'컬럼을 min-max 스케일 변환한 후, 상위 5%와 하위 5% 값의 합을 구하시오
'''
import pandas as pd
import numpy as np
df = pd.read_csv('basic1.csv')

from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import 
sc = MinMaxScaler()
df['f5'] = sc.fit_transform(df[['f5']])
upper = df['f5'].quantile(0.95)
lower = df['f5'].quantile(0.05)
print(upper+lower)
# print(sklearn.preprocessing.__all__)