'''
주어진 데이터 중 train.csv에서 'SalePrice'컬럼의 왜도와 첨도를 구한 값과, 'SalePrice'컬럼을 스케일링(log1p)로 
변환한 이후 왜도와 첨도를 구해 모두 더한 다음 소수점 2째자리까지 출력하시오
'''
import pandas as pd
import numpy as np

df = pd.read_csv('./h.csv')
b_skew = df['SalePrice'].skew()
b_kurt = df['SalePrice'].kurt()
print(b_skew,b_kurt)

df['SalePrice'] = np.log1p(df['SalePrice'])
a_skew = df['SalePrice'].skew()
a_kurt = df['SalePrice'].kurt()
print(a_skew,a_kurt)


print(round(b_skew+b_kurt+a_skew + a_kurt,2))
# print(dir(pd.DataFrame))