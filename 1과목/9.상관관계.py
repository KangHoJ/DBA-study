'''
상관관계 구하기 주어진 데이터에서 상관관계를 구하고, quality와의 상관관계가 가장 큰 값과, 가장 작은 값을 구한 다음 더하시오!
단, quality와 quality 상관관계 제외, 소수점 둘째 자리까지 출력
'''

import pandas as pd
import numpy as np
df = pd.read_csv('./wine.csv')

correlation = df.corr()
print(correlation.loc['quality'].sort_values(ascending=False).iloc[1:][0])
print(correlation.loc['quality'].sort_values(ascending=True)[0])