# 주어진 데이터에서 2022년 5월 주말과 평일의 sales컬럼 평균값 차이를 구하시오
import pandas as pd
import numpy as np
df = pd.read_csv('basic2.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['week'] = df['Date'].dt.dayofweek # 0~6까지 무슨 요일인지
a = df[(df['week']>=5) & (df['Date'].dt.year==2022) & (df['Date'].dt.month==5)]['Sales'].mean() # 주말
b = df[(df['week']<5) & (df['Date'].dt.year==2022) & (df['Date'].dt.month==5)]['Sales'].mean() # 평일
print(round(a-b,2))

df = pd.read_csv('basic2.csv',parse_dates=['Date']) # 이거로도 Date만들수있음