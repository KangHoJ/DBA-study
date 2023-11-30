'''
1.세션의 지속 시간을 분으로 계산하고 가장 긴 지속시간을 출력하시오(반올림 후 총 분만 출력)
2.가장 많이 머무른 Page를 찾고 그 페이지의 머문 평균 시간을 구하시오 (반올림 후 총 시간만 출력)
3.사용자들이 가장 활발히 활동하는 시간대(예: 새벽, 오전, 오후, 저녁)를 분석하세요. 이를 위해 하루를 4개의 시간대로 나누고 각 시간대별로 가장 많이 시작된 세션의 수를 계산하고, 그 중에 가장 많은 세션 수를 출력하시오

새벽: 0시 부터 6시 전
오전: 6시 부터 12시 전
오후: 12 부터 18시 전
저녁: 18시 부터 0시 전
4.user가 가장 많이 접속 했던 날짜를 출력하시오. (예, 2023-02-17)

'''

import pandas as pd
df = pd.read_csv("./website.csv")
print(df.info())

# 1.세션의 지속 시간을 분으로 계산하고 가장 긴 지속시간을 출력하시오(반올림 후 총 분만 출력)
df['StartTime'] = pd.to_datetime(df['StartTime'])
df['EndTime'] = pd.to_datetime(df['EndTime'])
df['during'] = df['EndTime']-df['StartTime']
df['during'] = df['during'].dt.seconds/60
print(df['during'].max())

# 2.가장 많이 머무른 Page를 찾고 그 페이지의 머문 평균 시간을 구하시오 (반올림 후 총 시간만 출력)
print(df['Page'].value_counts().idxmax()) # Page5
print(round(df[df['Page']=='Page5']['during'].mean()/60))

# 3.사용자들이 가장 활발히 활동하는 시간대(예: 새벽, 오전, 오후, 저녁)를 분석하세요. 이를 위해 하루를 4개의 시간대로 나누고 각 시간대별로 가장 많이 시작된 세션의 수를 계산하고, 그 중에 가장 많은 세션 수를 출력하시오
a = len(df[(df['StartTime'].dt.hour>=0) & (df['StartTime'].dt.hour<6)])
b = len(df[(df['StartTime'].dt.hour>=6) & (df['StartTime'].dt.hour<12)])
c = len(df[(df['StartTime'].dt.hour>=12) & (df['StartTime'].dt.hour<18)])
d = len(df[(df['StartTime'].dt.hour>=18) & (df['StartTime'].dt.hour<24)])
print(a,b,c,d)

#4.user가 가장 많이 접속 했던 날짜를 출력하시오. (예, 2023-02-17)
print(df['StartTime'].dt.date.value_counts().idxmax())
print(df['StartTime'].dt.date.value_counts().index[0])