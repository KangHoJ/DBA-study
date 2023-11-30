# city와 f4를 기준으로 f5의 평균값을 구한 다음, f5를 기준으로 상위 7개 값을 모두 더해 출력하시오 (소수점 둘째자리까지 출력)

import pandas as pd
import numpy as np
df = pd.read_csv('basic1.csv')

print(round(sum(df.groupby(['city','f4'])['f5'].mean().sort_values(ascending=False).head(7)),2))