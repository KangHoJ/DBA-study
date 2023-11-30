# 문제: 다음은 22명의 학생들이 국어시험에서 받은 점수이다. 학생들의 평균이 75보다 크다고 할 수 있는가?

# 귀무가설 : h0<75  / h1:m>75

from scipy import stats
scores = [75, 80, 68, 72, 77, 82, 81, 79, 70, 74, 76, 78, 81, 73, 81, 78, 75, 72, 74, 79, 78, 79]

mu=75
alpha = 0.05

t , p = stats.ttest_1samp(scores,mu,alternative='greater')
# two-sided(귀무가설: 첫번째 평균 = 두번째 평균 같다 )
# greater(귀무가설 : 첫번째 표본평균이 < 두번째 평균보다 작다)
# less(귀무가설: 첫번째 표본평균 > 두번째 평균보다 크다)

print(t,p)
if p < alpha:
    print('귀무가설을 기각합니다 , 모평균은 75보다 큽니다')
else:
    print("귀무가설을 채택합니다. 모평균은 75보다 크지 않습니다.")