## 图像显示中文的问题
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False
import seaborn as sns
sns.set(font= "Kaiti",style="ticks",font_scale=1.4)
import pandas as pd
pd.set_option("max_colwidth", 200)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,label_binarize
from sklearn.model_selection import  train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
## 忽略提醒
import warnings
warnings.filterwarnings("ignore")

# 导入数据，路径中要使用\\或者/或者在路径前面加r
dataset = pd.read_csv(r'C:\users\14903\Desktop\submitting\data\feature set 3.csv')

# 将字符串类型的分类变量重新编码
label = LabelEncoder()
dataset["GlucoseBMI"] = label.fit_transform(dataset["GlucoseBMI"])
dataset["AgeAge1"] = label.fit_transform(dataset["AgeAge1"])
dataset["AgeDiabetesPedigreeFunction1"] = label.fit_transform(dataset["AgeDiabetesPedigreeFunction1"])
dataset["GlucoseGlucose1"] = label.fit_transform(dataset["GlucoseGlucose1"])
dataset["GlucoseBloodPressure1"] = label.fit_transform(dataset["GlucoseBloodPressure1"])
dataset["BloodPressureBMI1"] = label.fit_transform(dataset["BloodPressureBMI1"])
dataset["BMIAge1"] = label.fit_transform(dataset["BMIAge1"])
dataset["BMIBMI1"] = label.fit_transform(dataset["BMIBMI1"])
dataset["Outcome3"] = label.fit_transform(dataset["Outcome3"])

# 定于预测目标变量
target = ["Outcome3"]

# 定义模型的自变量
train_x = ["GlucoseBMI","AgeAge1","AgeDiabetesPedigreeFunction1","GlucoseGlucose1",
           "GlucoseBloodPressure1","BloodPressureBMI1","BMIAge1","BMIBMI1"]

# 将数据集切分为训练集和测试集
X_train,X_test,Y_train,Y_test = train_test_split(dataset[train_x],dataset[target],
                                                 test_size=0.3,random_state=123)  # random_state 是随机数种子

# 使用随机森林对数据集进行分类
MLP1 = MLPClassifier(
                      random_state=63)
MLP1.fit(X_train,Y_train)

# 输出其在训练集和测试集上的预测精度
RFC1_train = MLP1.predict(X_train)
RFC1_test = MLP1.predict(X_test)
print("训练数据集上的精度:",accuracy_score(Y_train,RFC1_train))
print("验证数据集上的精度:",accuracy_score(Y_test,RFC1_test))

# 可视化在测试集上的ROC曲线
pre_test = MLP1.predict_proba(X_test)[:,1]
FPR_NB,TPR_NB,_ = roc_curve(Y_test,pre_test)
aucval = auc(FPR_NB,TPR_NB)  # 计算auc的取值
plt.figure(figsize=(8,8))
plt.plot([0,1],[0,1],"k--")
plt.plot(FPR_NB, TPR_NB,"r",linewidth = 3)
plt.grid()
plt.xlabel("假正率")
plt.ylabel("真正率")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("随机森林ROC曲线")
plt.text(0.05,0.9,"AUC = "+str(round(aucval,2)))
plt.show()

