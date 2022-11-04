"""
Figure 12: ROC curves of feature set 3
"""


# 图像显示中文的问题
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set(font="Kaiti", style="ticks", font_scale=1.4)
import pandas as pd
pd.set_option("max_colwidth", 200)
# 忽略提醒
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

def k_auc_figure():
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 3.csv')

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
    train_x = ["GlucoseBMI", "AgeAge1", "AgeDiabetesPedigreeFunction1", "GlucoseGlucose1",
               "GlucoseBloodPressure1", "BloodPressureBMI1", "BMIAge1", "BMIBMI1"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用随机森林对数据集进行分类
    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=16
    )
    RFC1.fit(X_train, Y_train)

    # 使用神经网络对数据集进行分类
    MLP1 = MLPClassifier(
        random_state=63
    )
    MLP1.fit(X_train, Y_train)

    # 使用逻辑回归对数据集进行分类
    IRC1 = LogisticRegression(
        random_state=16
    )
    IRC1.fit(X_train, Y_train)

    # 使用AdaBoost对数据集进行分类
    ABC1 = AdaBoostClassifier(
        random_state=16
    )
    ABC1.fit(X_train, Y_train)

    # 可视化在测试集上的ROC曲线
    pre_test1 = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB1, TPR_NB1, _ = roc_curve(Y_test, pre_test1)
    aucval1 = auc(FPR_NB1, TPR_NB1)  # 计算auc的取值

    pre_test2 = MLP1.predict_proba(X_test)[:, 1]
    FPR_NB2, TPR_NB2, _ = roc_curve(Y_test, pre_test2)
    aucval2 = auc(FPR_NB2, TPR_NB2)  # 计算auc的取值

    pre_test3 = IRC1.predict_proba(X_test)[:, 1]
    FPR_NB3, TPR_NB3, _ = roc_curve(Y_test, pre_test3)
    aucval3 = auc(FPR_NB3, TPR_NB3)  # 计算auc的取值

    pre_test4 = ABC1.predict_proba(X_test)[:, 1]
    FPR_NB4, TPR_NB4, _ = roc_curve(Y_test, pre_test4)
    aucval4 = auc(FPR_NB4, TPR_NB4)  # 计算auc的取值

    plt.figure(figsize=(8, 8), dpi=120)
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB1, TPR_NB1, "r", linewidth=3, label="Model 3(feature set 3 + RF)")
    plt.plot(FPR_NB2, TPR_NB2, "b", linewidth=3, label="Model 10(feature set 3 + ANN)")
    plt.plot(FPR_NB3, TPR_NB3, "g", linewidth=3, label="Model 14(feature set 3 + LR)")
    plt.plot(FPR_NB4, TPR_NB4, "black", linewidth=3, label="Model 18(feature set 3 + AdaBoost)")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("ROC curves of various algorithms for feature set 3")
    plt.text(0.65, 0.6, "AUC = " + str(round(aucval1, 2)), fontdict={"size": "20", "color": "r"})
    plt.text(0.65, 0.55, "AUC = " + str(round(aucval2, 2)), fontdict={"size": "20", "color": "b"})
    plt.text(0.65, 0.5, "AUC = " + str(round(aucval3, 2)), fontdict={"size": "20", "color": "g"})
    plt.text(0.65, 0.45, "AUC = " + str(round(aucval4, 2)), fontdict={"size": "20", "color": "black"})
    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':

    k_auc_figure()