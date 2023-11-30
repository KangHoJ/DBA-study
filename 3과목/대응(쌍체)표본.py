# 한사람이 두가지 다른 실험
# 주어진 데이터는 고혈압 환자 치료 전후의 혈압이다. 해당 치료가 효과가 있는지 대응(쌍체)표본 t-검정을 진행하시오

# 귀무가설 :m : b - a >=0(혈압높음) -> 효과없다 , 대립가설: m : b - a <0 -> (효과있다)

import pandas as pd
from spicy import stats
df = pd.read_csv('./data/high_blood_pressure.csv')
print(df.head())

# 표본평균(m)은 ?
df['diff'] = df['bp_post'] - df['bp_pre']
round(df['diff'].mean(),2)

# 검정통계랑 값은?
st , pv = stats.ttest_rel(df['bp_post'],df['bp_pre'],alternative='less')
print(round(st,4))
# two-sided(귀무가설: 첫번째 평균 = 두번째 평균 같다 )
# greater(귀무가설 : 첫번째 표본평균이 < 두번째 평균보다 작다)
# less(귀무가설: 첫번째 표본평균 > 두번째 평균보다 크다)



# p-값은 ?
print(round(pv,4))

# 가설검정 결과는 ?(유의수준 5%)