import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e6_p1_1.csv')
df.head(5)

'''각 구급 보고서 별 출동시각과 신고시각의 차이를 ‘소요시간’ 컬럼을 만들고 초(sec)단위로 구하고
 소방서명 별 소요시간의 평균을 오름차순으로 정렬 했을때 3번째로 작은 소요시간의 값과 소방서명을 출력하라'''

# print(df['소요시간'])
# df['신고시각'] = pd.to_datetime(df['신고시각'],format='%')
# df['출동시각'] = pd.to_datetime(df['출동시각'])
# df['소요시간'] = pd.to_datetime(df['소요시간'])
# df['소요시간'] = df['출동시각'] - df['신고시각']
# print(df['소요시간'])

df['소요시각'] = (pd.to_datetime(df['출동일자'].astype('str') + df['출동시각'].astype('str').str.zfill(6)) -
pd.to_datetime(df['신고일자'].astype('str') + df['신고시각'].astype('str').str.zfill(6))).dt.total_seconds()
result = df.groupby('소방서명')['소요시각'].mean().sort_values(ascending=True)
print(result.iloc[2])

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e6_p1_2.csv')
df.head(5)

# 학교 세부유형이 일반중학교인 학교들 중 일반중학교 숫자가 2번째로 많은 시도의 일반중학교 데이터만 필터하여 
# 해당 시도의 교원 한명 당 맡은 학생수가 가장 많은 학교를 찾아서 해당 학교의 교원수를 출력하라
df = df[(df['학교세부유형']=='일반중학교') & (df['시도']=='서울')]
df['맡은수'] = df['일반학급_학생수_계'] / df['교원수_총계_계']
print(df.sort_values('맡은수',ascending=False)) # 33


