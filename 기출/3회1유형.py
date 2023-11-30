import pandas as pd
import numpy as np

#  1-1 ) 2022년 데이터 중 2022년 중앙값보다 큰 값의 데이터 수
df = pd.read_csv('t1-data2.csv',index_col='year')
df_2022 = df[df.index=='2022년']
df_new = df_2022[(df_2022>54.5)]
# print(sum(df_new.count()))

#풀이2
# print(sum(df.loc['2022년',:]>54.5))

# df = pd.read_csv('t1-data1.csv')
# print(df.isnull().sum() , df.shape)
# df = df.dropna(subset='f1',axis=0)
# print(df.isnull().sum() , df.shape)
# per = df.shape[0] * 0.6
# print(per) # 36
# df = df.head(36)
# print(df['f1'].quantile(0.75))

#풀이2
# df = df.iloc[:(int(len(df)*0.6))]
# print(df['f1'].quantile(0.75))

#풀이3
df = pd.read_csv('t1-data1.csv')
print(df.isnull().sum().idxmax())
print(df.isnull().sum().index[3])
print(df.isnull().sum().values[3])
print(df.isnull().sum().max())