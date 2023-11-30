# 주어진 데이터에서 상위 10개 국가의 접종률 평균과 하위 10개 국가의 접종률 평균을 구하고, 그 차이를 구해보세요 
import pandas as pd
import numpy as np
df = pd.read_csv('./covid.csv')

# print(df.head())
df = df[df['ratio']<=100]
print(np.mean(df.groupby('country')['ratio'].max().sort_values(ascending=False).iloc[0:10].values))
print(np.mean(df.groupby('country')['ratio'].max().sort_values(ascending=True).iloc[0:10].values))