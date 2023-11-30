import	pandas	as	pd
import	numpy	as	np
#	실기 시험 데이터셋으로 셋팅하기	(수정금지)
from	sklearn.datasets	import	load_diabetes
#	diabetes	데이터셋 로드
diabetes	=	load_diabetes()
x	=	pd.DataFrame(diabetes.data,	columns=diabetes.feature_names)
y	=	pd.DataFrame(diabetes.target)
y.columns	= ['target']


## 귀무가설 : 두 변수간 선형관계가 존재 x 
x = x['bmi']
y = y['target']

from scipy import stats
r , p = stats.pearsonr(x,y)
print(r,p) # 선형관계 존재한다 , r은 상관계수


