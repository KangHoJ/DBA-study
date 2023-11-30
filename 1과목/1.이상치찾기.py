### 주어진 데이터에서 age 이상치(소수점 나이)를 찾고 올림, 내림, 버림(절사)했을때 3가지 모두 이상치 'age' 평균을 구한 다음 모두 더하여 출력하시오
import pandas as pd
import numpy as np
df = pd.read_csv('./basic1.csv')

# 이상치(소수점 나이)
out = df[df['age']-np.floor(df['age'])!=0] # 소수점
out = df[(df['age'] == round(df['age'],0))] # 방법2

ceil = np.ceil(out['age']).mean()
floor = np.floor(out['age']).mean()
trunc = np.trunc(out['age']).mean()
print(ceil+floor+trunc)
ㅇㅇ