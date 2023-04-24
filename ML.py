import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# 讀取資料並轉換為numpy.ndarray格式
data = pd.read_csv("C:/Users/CNN/Downloads/ShauYuYan/S/645_NonS_feature.csv")

# 進行5-fold交叉驗證
scores = cross_val_score(#network
                        #estimator = MLPClassifier(), 
                        #estimator = LogisticRegression,
                        #estimator = KNeighborsClassifier,
                         estimator = SVC(kernel='rbf',probability=True),
                         #feature
                         X = data.iloc[1:, 1:].values,
                         #label
                         y = data.iloc[1:, 0].values,
                         #5-fold  
                         cv=5                  
                         )
# 輸出5次準確率分數和平均準確率分數
print('Scores:', scores)
print('Mean Accuracy:', scores.mean())