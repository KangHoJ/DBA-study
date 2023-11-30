# 2022년 5월 sales의 중앙값을 구하시오
import pandas as pd
import numpy as np
df = pd.read_csv('basic2.csv')

# print(df.head())
# print(df.info())

df['Date'] = pd.to_datetime(df['Date'])
print(df[(df['Date'].dt.year==2022) & (df['Date'].dt.month==5)]['Sales'].median())

df = pd.read_csv('basic2.csv',parse_dates=['Date']) # 이거로도 Date만들수있음