import pandas as pd
import numpy as np
from spicy import stats

# 문제 mpg열 평균이 20과 같다고 할수있는지 검정 (단일표본검정)
df = pd.read_csv('./mtcars.csv')

# 가설 (귀무가설 : m=20)
# print(dir(stats))

# 정규성 검정
result = stats.shapiro(df['mpg']) # p-value가 0.05보다 크므로 정규성 만족
print(result)

result = stats.wilcoxon(df['mpg']-20,alternative='two-sided') # 만약 정규성 만족 안한다면 해야함
print(result)

result = stats.ttest_1samp(df['mpg'],20,alternative='two-sided') # p 0.05보다 크므로 같다고 할수있다.
print(result)
