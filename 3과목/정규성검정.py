# 12명의 수험생이 빅데이터 분석기사 시험에서 받은 점수이다. Shapiro-Wilk 검정을 사용하여 데이터가 정규 분포를 따르는지 검증하시오
# 귀무가설 : 정규 분포를 따른다 / 대립가설 : 정규 분포를 따르지 않는다 

from spicy import stats
data = [75, 83, 81, 92, 68, 77, 78, 80, 85, 95, 79, 89]

s,p = stats.shapiro(data)
print('검정통계량',s)
print('p',p)
alpha = 0.5

if p<alpha :
    print('정규분포를 따르지 않는다 ')
else:
    print('정규분포를 따른다')