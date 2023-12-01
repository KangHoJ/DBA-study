'''
주어진 데이터에서 2022년 월별 Sales 합계 중 가장 큰 금액과 2023년 월별 Sales 합계 중 가장 큰 금액의 차이를 절대값으로 구하시오.
단 Events컬럼이 '1'인경우 80%의 Salse값만 반영함
'''
import pandas as pd
import numpy as np
df = pd.read_csv('basic2.csv')
# print(df.info())
# df = np.where(df['Events']==1,df['Sales']=df['Sales']*0.8,df['Sales']=df['Sales'])
def event_sales(x):
    if x['Events'] == 1:
        x['Sales'] = x['Sales']*0.8
    else:
        x['Sales'] = x['Sales']
    return x
df = df.apply(lambda x : event_sales(x) , axis=1)
# print(df)


df['Date'] = pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month
df_2022 = df[df['Date'].dt.year == 2022]
df_2023 = df[df['Date'].dt.year == 2023]

a = df_2022.groupby('month')['Sales'].sum().max()
b = df_2023.groupby('month')['Sales'].sum().max()
print(round(abs(a-b),2))

