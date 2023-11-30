import pandas as pd
import numpy as np
from scipy import stats


# 문제 : 랜덤 박스에 상품	A,B,C,D가 들어있다. 다음은 랜덤박스에서	100번 상품을 꺼냈을 때의 상품 데이터라고 할 때 상품이 동일한 비율로 들어있다고 할 수 있는지 검정
# 귀무가설 : 동일한 비율로 들어있다

row1	= [30, 20, 15, 35]
df=pd.DataFrame([row1],	columns=['A','B','C', 'D'])

f_obs = [30,20,15,35]
f_exp = [25,25,25,25]

result = stats.chisquare(f_obs=f_obs,f_exp=f_exp)
print(result)

# 문제 : A 30%,	B 15%,	C 55%	비율로 들어있다고 할 수 있는지 검정해보시오

row1	= [50,25,75]
df	=	pd.DataFrame([row1],	columns=['A','B','C'])
df

f_obs = [50,25,75]
f_exp = [150*0.3,150*0.15,150*0.55]
result = stats.chisquare(f_obs=f_obs,f_exp=f_exp)
print(result)

