import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)

correation = df.corr()
print(round(correation.loc['sepal length (cm)', 'sepal width (cm)'],2)) 
# 두개만 뽑으려면 다음과 같이 loc로
