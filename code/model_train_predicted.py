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
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import *


# 随机森林：768原始数据
def RFC1():
    """随机森林：768原始数据"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 1.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["Pregnancies"] = label.fit_transform(dataset["Pregnancies"])
    dataset["Glucose"] = label.fit_transform(dataset["Glucose"])
    dataset["BloodPressure"] = label.fit_transform(dataset["BloodPressure"])
    dataset["SkinThickness"] = label.fit_transform(dataset["SkinThickness"])
    dataset["Insulin"] = label.fit_transform(dataset["Insulin"])
    dataset["BMI"] = label.fit_transform(dataset["BMI"])
    dataset["DiabetesPedigreeFunction"] = label.fit_transform(dataset["DiabetesPedigreeFunction"])
    dataset["Age"] = label.fit_transform(dataset["Age"])
    dataset["Outcome"] = label.fit_transform(dataset["Outcome"])

    # 定于预测目标变量
    target = ["Outcome"]

    # 定义模型的自变量
    train_x = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
               "Age"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用随机森林对数据集进行分类
    RFC1 = RandomForestClassifier(
                                    oob_score=True,
                                    random_state=16
                                )
    RFC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)
    print("随机森林的OOB score:", RFC1.oob_score_)
    print("训练数据集上的精度:", accuracy_score(Y_train, RFC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, RFC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("随机森林ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)


# 随机森林：768最大相关最小冗余
def RFC2():
    """随机森林：768最大相关最小冗余"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 2.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["GlucoseBMI"] = label.fit_transform(dataset["GlucoseBMI"])
    dataset["PregnanciesDiabetesPedigreeFunction1"] = label.fit_transform(
        dataset["PregnanciesDiabetesPedigreeFunction1"])
    dataset["PregnanciesAge1"] = label.fit_transform(dataset["PregnanciesAge1"])
    dataset["GlucoseGlucose1"] = label.fit_transform(dataset["GlucoseGlucose1"])
    dataset["GlucoseBloodPressure1"] = label.fit_transform(dataset["GlucoseBloodPressure1"])
    dataset["BloodPressureBMI1"] = label.fit_transform(dataset["BloodPressureBMI1"])
    dataset["SkinThicknessBMI1"] = label.fit_transform(dataset["SkinThicknessBMI1"])
    dataset["SkinThicknessAge1"] = label.fit_transform(dataset["SkinThicknessAge1"])
    dataset["Outcome1"] = label.fit_transform(dataset["Outcome1"])

    # 定于预测目标变量
    target = ["Outcome1"]

    # 定义模型的自变量
    train_x = ["GlucoseBMI", "PregnanciesDiabetesPedigreeFunction1", "PregnanciesAge1", "GlucoseGlucose1",
               "GlucoseBloodPressure1", "BloodPressureBMI1", "SkinThicknessBMI1", "SkinThicknessAge1"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用随机森林对数据集进行分类
    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=16
    )
    RFC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)
    print("随机森林的OOB score:", RFC1.oob_score_)
    print("训练数据集上的精度:", accuracy_score(Y_train, RFC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, RFC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("随机森林ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)


# 随机森林：768关键特征
def RFC3():
    """随机森林：768关键特征"""
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

    # 输出其在训练集和测试集上的预测精度
    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)
    print("随机森林的OOB score:", RFC1.oob_score_)
    print("训练数据集上的精度:", accuracy_score(Y_train, RFC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, RFC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("随机森林ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)


# 随机森林：2000原始数据
def RFC4():
    """随机森林：2000原始数据"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 4.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["Pregnancies00"] = label.fit_transform(dataset["Pregnancies00"])
    dataset["Glucose00"] = label.fit_transform(dataset["Glucose00"])
    dataset["BloodPressure00"] = label.fit_transform(dataset["BloodPressure00"])
    dataset["SkinThickness00"] = label.fit_transform(dataset["SkinThickness00"])
    dataset["Insulin00"] = label.fit_transform(dataset["Insulin00"])
    dataset["BMI00"] = label.fit_transform(dataset["BMI00"])
    dataset["DiabetesPedigreeFunction00"] = label.fit_transform(dataset["DiabetesPedigreeFunction00"])
    dataset["Age00"] = label.fit_transform(dataset["Age00"])
    dataset["Outcome00"] = label.fit_transform(dataset["Outcome00"])

    # 定于预测目标变量
    target = ["Outcome00"]

    # 定义模型的自变量
    train_x = ["Pregnancies00", "Glucose00", "BloodPressure00", "SkinThickness00",
               "Insulin00", "BMI00", "DiabetesPedigreeFunction00", "Age00"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用随机森林对数据集进行分类
    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=22)
    RFC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)
    print("随机森林的OOB score:", RFC1.oob_score_)
    print("训练数据集上的精度:", accuracy_score(Y_train, RFC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, RFC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("随机森林ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)


# 随机森林：2000最大相关最小冗余
def RFC5():
    """随机森林：2000最大相关最小冗余"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 5.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["GlucoseAge00"] = label.fit_transform(dataset["GlucoseAge00"])
    dataset["BMIDiabetesPedigreeFunction00"] = label.fit_transform(dataset["BMIDiabetesPedigreeFunction00"])
    dataset["PregnanciesSkinThickness11"] = label.fit_transform(dataset["PregnanciesSkinThickness11"])
    dataset["GlucoseGlucose11"] = label.fit_transform(dataset["GlucoseGlucose11"])
    dataset["GlucoseBloodPressure11"] = label.fit_transform(dataset["GlucoseBloodPressure11"])
    dataset["GlucoseBMI11"] = label.fit_transform(dataset["GlucoseBMI11"])
    dataset["BloodPressureBMI11"] = label.fit_transform(dataset["BloodPressureBMI11"])
    dataset["DiabetesPedigreeFunctionAge11"] = label.fit_transform(dataset["DiabetesPedigreeFunctionAge11"])
    dataset["Outcome11"] = label.fit_transform(dataset["Outcome11"])

    # 定于预测目标变量
    target = ["Outcome11"]

    # 定义模型的自变量
    train_x = ["GlucoseAge00", "BMIDiabetesPedigreeFunction00", "PregnanciesSkinThickness11", "GlucoseGlucose11",
               "GlucoseBloodPressure11", "GlucoseBMI11", "BloodPressureBMI11", "DiabetesPedigreeFunctionAge11"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用随机森林对数据集进行分类
    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=22)
    RFC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)
    print("随机森林的OOB score:", RFC1.oob_score_)
    print("训练数据集上的精度:", accuracy_score(Y_train, RFC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, RFC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("随机森林ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)


# 随机森林：2000关键特征
def RFC6():
    """随机森林：2000关键特征"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 6.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["GlucoseAge00"] = label.fit_transform(dataset["GlucoseAge00"])
    dataset["BMIDiabetesPedigreeFunction00"] = label.fit_transform(dataset["BMIDiabetesPedigreeFunction00"])
    dataset["AgeSkinThickness11"] = label.fit_transform(dataset["AgeSkinThickness11"])
    dataset["GlucoseGlucose11"] = label.fit_transform(dataset["GlucoseGlucose11"])
    dataset["GlucoseBloodPressure11"] = label.fit_transform(dataset["GlucoseBloodPressure11"])
    dataset["GlucoseBMI11"] = label.fit_transform(dataset["GlucoseBMI11"])
    dataset["BloodPressureBMI11"] = label.fit_transform(dataset["BloodPressureBMI11"])
    dataset["DiabetesPedigreeFunctionAge11"] = label.fit_transform(dataset["DiabetesPedigreeFunctionAge11"])
    dataset["Outcome33"] = label.fit_transform(dataset["Outcome33"])

    # 定于预测目标变量
    target = ["Outcome33"]

    # 定义模型的自变量
    train_x = ["GlucoseAge00", "BMIDiabetesPedigreeFunction00", "AgeSkinThickness11", "GlucoseGlucose11",
               "GlucoseBloodPressure11", "GlucoseBMI11", "BloodPressureBMI11", "DiabetesPedigreeFunctionAge11"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用随机森林对数据集进行分类
    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=22)
    RFC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)
    print("随机森林的OOB score:", RFC1.oob_score_)
    print("训练数据集上的精度:", accuracy_score(Y_train, RFC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, RFC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("随机森林ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()
    feat_important = RFC1.feature_importances_
    print(feat_important)


# 随机森林：768 因果置换
def RFC7():
    """随机森林：768 因果置换"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 7.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["GlucoseSkinThickness"] = label.fit_transform(dataset["GlucoseSkinThickness"])
    dataset["PregnanciesDiabetesPedigreeFunction1"] = label.fit_transform(
        dataset["PregnanciesDiabetesPedigreeFunction1"])
    dataset["PregnanciesPregnancies1"] = label.fit_transform(dataset["PregnanciesPregnancies1"])
    dataset["GlucoseGlucose1"] = label.fit_transform(dataset["GlucoseGlucose1"])
    dataset["GlucoseBloodPressure1"] = label.fit_transform(dataset["GlucoseBloodPressure1"])
    dataset["BloodPressureSkinThickness1"] = label.fit_transform(dataset["BloodPressureSkinThickness1"])
    dataset["SkinThicknessSkinThickness1"] = label.fit_transform(dataset["SkinThicknessSkinThickness1"])
    dataset["SkinThicknessPregnancies1"] = label.fit_transform(dataset["SkinThicknessPregnancies1"])
    dataset["Outcome5"] = label.fit_transform(dataset["Outcome5"])

    # 定于预测目标变量
    target = ["Outcome5"]

    # 定义模型的自变量
    train_x = ["GlucoseSkinThickness", "PregnanciesDiabetesPedigreeFunction1", "PregnanciesPregnancies1",
               "GlucoseGlucose1", "GlucoseBloodPressure1", "BloodPressureSkinThickness1",
               "SkinThicknessSkinThickness1", "SkinThicknessPregnancies1"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用随机森林对数据集进行分类
    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=16)
    RFC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)
    print("随机森林的OOB score:", RFC1.oob_score_)
    print("训练数据集上的精度:", accuracy_score(Y_train, RFC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, RFC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("随机森林ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()
    feat_important = RFC1.feature_importances_
    print(feat_important)


# 随机森林：2000 因果置换
def RFC8():
    """随机森林：2000 因果置换"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 8.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["PregnanciesGlucose00"] = label.fit_transform(dataset["PregnanciesGlucose00"])
    dataset["SkinThicknessDiabetesPedigreeFunction00"] = label.fit_transform(
        dataset["SkinThicknessDiabetesPedigreeFunction00"])
    dataset["PregnanciesSkinThickness11"] = label.fit_transform(dataset["PregnanciesSkinThickness11"])
    dataset["GlucoseGlucose11"] = label.fit_transform(dataset["GlucoseGlucose11"])
    dataset["GlucoseBloodPressure11"] = label.fit_transform(dataset["GlucoseBloodPressure11"])
    dataset["GlucoseSkinThickness11"] = label.fit_transform(dataset["GlucoseSkinThickness11"])
    dataset["BloodPressureSkinThickness11"] = label.fit_transform(dataset["BloodPressureSkinThickness11"])
    dataset["DiabetesPedigreeFunctionPregnancies11"] = label.fit_transform(
        dataset["DiabetesPedigreeFunctionPregnancies11"])
    dataset["Outcome55"] = label.fit_transform(dataset["Outcome55"])

    # 定于预测目标变量
    target = ["Outcome55"]

    # 定义模型的自变量
    train_x = ["PregnanciesGlucose00", "SkinThicknessDiabetesPedigreeFunction00", "PregnanciesSkinThickness11",
               "GlucoseGlucose11", "GlucoseBloodPressure11", "GlucoseSkinThickness11",
               "BloodPressureSkinThickness11", "DiabetesPedigreeFunctionPregnancies11"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用随机森林对数据集进行分类
    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=22)
    RFC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)
    print("随机森林的OOB score:", RFC1.oob_score_)
    print("训练数据集上的精度:", accuracy_score(Y_train, RFC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, RFC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("随机森林ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()
    feat_important = RFC1.feature_importances_
    print(feat_important)


# 神经网络：768原始数据
def NNC1():
    """神经网络：768原始数据"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 1.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["Pregnancies"] = label.fit_transform(dataset["Pregnancies"])
    dataset["Glucose"] = label.fit_transform(dataset["Glucose"])
    dataset["BloodPressure"] = label.fit_transform(dataset["BloodPressure"])
    dataset["SkinThickness"] = label.fit_transform(dataset["SkinThickness"])
    dataset["Insulin"] = label.fit_transform(dataset["Insulin"])
    dataset["BMI"] = label.fit_transform(dataset["BMI"])
    dataset["DiabetesPedigreeFunction"] = label.fit_transform(dataset["DiabetesPedigreeFunction"])
    dataset["Age"] = label.fit_transform(dataset["Age"])
    dataset["Outcome"] = label.fit_transform(dataset["Outcome"])

    # 定于预测目标变量
    target = ["Outcome"]

    # 定义模型的自变量
    train_x = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
               "Age"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用神经网络对数据集进行分类
    MLP1 = MLPClassifier(
                            random_state=63
                        )
    MLP1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    MLP1_train = MLP1.predict(X_train)
    MLP1_test = MLP1.predict(X_test)
    print("训练数据集上的精度:", accuracy_score(Y_train, MLP1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, MLP1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = MLP1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("神经网络ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


# 神经网络：768关键特征
def NNC2():
    """神经网络：768关键特征"""
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
    MLP1 = MLPClassifier(
        random_state=63
    )
    MLP1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    MLP1_train = MLP1.predict(X_train)
    MLP1_test = MLP1.predict(X_test)
    print("训练数据集上的精度:", accuracy_score(Y_train, MLP1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, MLP1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = MLP1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("神经网络ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


# 神经网络：2000原始数据
def NNC3():
    """神经网络：2000原始数据"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 4.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["Pregnancies00"] = label.fit_transform(dataset["Pregnancies00"])
    dataset["Glucose00"] = label.fit_transform(dataset["Glucose00"])
    dataset["BloodPressure00"] = label.fit_transform(dataset["BloodPressure00"])
    dataset["SkinThickness00"] = label.fit_transform(dataset["SkinThickness00"])
    dataset["Insulin00"] = label.fit_transform(dataset["Insulin00"])
    dataset["BMI00"] = label.fit_transform(dataset["BMI00"])
    dataset["DiabetesPedigreeFunction00"] = label.fit_transform(dataset["DiabetesPedigreeFunction00"])
    dataset["Age00"] = label.fit_transform(dataset["Age00"])
    dataset["Outcome00"] = label.fit_transform(dataset["Outcome00"])

    # 定于预测目标变量
    target = ["Outcome00"]

    # 定义模型的自变量
    train_x = ["Pregnancies00", "Glucose00", "BloodPressure00", "SkinThickness00",
               "Insulin00", "BMI00", "DiabetesPedigreeFunction00", "Age00"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用随机森林对数据集进行分类
    MLP1 = MLPClassifier(
        random_state=16
    )
    MLP1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    MLP1_train = MLP1.predict(X_train)
    MLP1_test = MLP1.predict(X_test)
    print("训练数据集上的精度:", accuracy_score(Y_train, MLP1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, MLP1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = MLP1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("神经网络ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


# 神经网络：2000关键特征
def NNC4():
    """神经网络：2000关键特征"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 6.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["GlucoseAge00"] = label.fit_transform(dataset["GlucoseAge00"])
    dataset["BMIDiabetesPedigreeFunction00"] = label.fit_transform(dataset["BMIDiabetesPedigreeFunction00"])
    dataset["AgeSkinThickness11"] = label.fit_transform(dataset["AgeSkinThickness11"])
    dataset["GlucoseGlucose11"] = label.fit_transform(dataset["GlucoseGlucose11"])
    dataset["GlucoseBloodPressure11"] = label.fit_transform(dataset["GlucoseBloodPressure11"])
    dataset["GlucoseBMI11"] = label.fit_transform(dataset["GlucoseBMI11"])
    dataset["BloodPressureBMI11"] = label.fit_transform(dataset["BloodPressureBMI11"])
    dataset["DiabetesPedigreeFunctionAge11"] = label.fit_transform(dataset["DiabetesPedigreeFunctionAge11"])
    dataset["Outcome33"] = label.fit_transform(dataset["Outcome33"])

    # 定于预测目标变量
    target = ["Outcome33"]

    # 定义模型的自变量
    train_x = ["GlucoseAge00", "BMIDiabetesPedigreeFunction00", "AgeSkinThickness11", "GlucoseGlucose11",
               "GlucoseBloodPressure11", "GlucoseBMI11", "BloodPressureBMI11", "DiabetesPedigreeFunctionAge11"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用随机森林对数据集进行分类
    MLP1 = MLPClassifier(
        random_state=16
    )
    MLP1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    MLP1_train = MLP1.predict(X_train)
    MLP1_test = MLP1.predict(X_test)
    print("训练数据集上的精度:", accuracy_score(Y_train, MLP1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, MLP1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = MLP1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("神经网络ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()



# 逻辑回归：768原始数据
def IRC1():
    """逻辑回归：768原始数据"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 1.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["Pregnancies"] = label.fit_transform(dataset["Pregnancies"])
    dataset["Glucose"] = label.fit_transform(dataset["Glucose"])
    dataset["BloodPressure"] = label.fit_transform(dataset["BloodPressure"])
    dataset["SkinThickness"] = label.fit_transform(dataset["SkinThickness"])
    dataset["Insulin"] = label.fit_transform(dataset["Insulin"])
    dataset["BMI"] = label.fit_transform(dataset["BMI"])
    dataset["DiabetesPedigreeFunction"] = label.fit_transform(dataset["DiabetesPedigreeFunction"])
    dataset["Age"] = label.fit_transform(dataset["Age"])
    dataset["Outcome"] = label.fit_transform(dataset["Outcome"])

    # 定于预测目标变量
    target = ["Outcome"]

    # 定义模型的自变量
    train_x = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
               "Age"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用逻辑回归对数据集进行分类
    IRC1 = LogisticRegression(random_state=16)
    IRC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    IRC1_train = IRC1.predict(X_train)
    IRC1_test = IRC1.predict(X_test)

    print("训练数据集上的精度:", accuracy_score(Y_train, IRC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, IRC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = IRC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("逻辑回归ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


# 逻辑回归：768关键特征
def IRC2():
    """逻辑回归：768关键特征"""
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

    # 使用逻辑回归对数据集进行分类
    IRC1 = LogisticRegression(random_state=16)
    IRC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    IRC1_train = IRC1.predict(X_train)
    IRC1_test = IRC1.predict(X_test)

    print("训练数据集上的精度:", accuracy_score(Y_train, IRC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, IRC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = IRC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("逻辑回归ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


# 逻辑回归：2000原始数据
def IRC3():
    """逻辑回归：2000原始数据"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 4.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["Pregnancies00"] = label.fit_transform(dataset["Pregnancies00"])
    dataset["Glucose00"] = label.fit_transform(dataset["Glucose00"])
    dataset["BloodPressure00"] = label.fit_transform(dataset["BloodPressure00"])
    dataset["SkinThickness00"] = label.fit_transform(dataset["SkinThickness00"])
    dataset["Insulin00"] = label.fit_transform(dataset["Insulin00"])
    dataset["BMI00"] = label.fit_transform(dataset["BMI00"])
    dataset["DiabetesPedigreeFunction00"] = label.fit_transform(dataset["DiabetesPedigreeFunction00"])
    dataset["Age00"] = label.fit_transform(dataset["Age00"])
    dataset["Outcome00"] = label.fit_transform(dataset["Outcome00"])

    # 定于预测目标变量
    target = ["Outcome00"]

    # 定义模型的自变量
    train_x = ["Pregnancies00", "Glucose00", "BloodPressure00", "SkinThickness00",
               "Insulin00", "BMI00", "DiabetesPedigreeFunction00", "Age00"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用逻辑回归对数据集进行分类
    IRC1 = LogisticRegression(random_state=16)
    IRC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    IRC1_train = IRC1.predict(X_train)
    IRC1_test = IRC1.predict(X_test)

    print("训练数据集上的精度:", accuracy_score(Y_train, IRC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, IRC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = IRC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("逻辑回归ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


# 逻辑回归：2000关键特征
def IRC4():
    """逻辑回归：2000关键特征"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 6.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["GlucoseAge00"] = label.fit_transform(dataset["GlucoseAge00"])
    dataset["BMIDiabetesPedigreeFunction00"] = label.fit_transform(dataset["BMIDiabetesPedigreeFunction00"])
    dataset["AgeSkinThickness11"] = label.fit_transform(dataset["AgeSkinThickness11"])
    dataset["GlucoseGlucose11"] = label.fit_transform(dataset["GlucoseGlucose11"])
    dataset["GlucoseBloodPressure11"] = label.fit_transform(dataset["GlucoseBloodPressure11"])
    dataset["GlucoseBMI11"] = label.fit_transform(dataset["GlucoseBMI11"])
    dataset["BloodPressureBMI11"] = label.fit_transform(dataset["BloodPressureBMI11"])
    dataset["DiabetesPedigreeFunctionAge11"] = label.fit_transform(dataset["DiabetesPedigreeFunctionAge11"])
    dataset["Outcome33"] = label.fit_transform(dataset["Outcome33"])

    # 定于预测目标变量
    target = ["Outcome33"]

    # 定义模型的自变量
    train_x = ["GlucoseAge00", "BMIDiabetesPedigreeFunction00", "AgeSkinThickness11", "GlucoseGlucose11",
               "GlucoseBloodPressure11", "GlucoseBMI11", "BloodPressureBMI11", "DiabetesPedigreeFunctionAge11"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用逻辑回归对数据集进行分类
    IRC1 = LogisticRegression(random_state=16)
    IRC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    IRC1_train = IRC1.predict(X_train)
    IRC1_test = IRC1.predict(X_test)

    print("训练数据集上的精度:", accuracy_score(Y_train, IRC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, IRC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = IRC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("逻辑回归ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


# AdaBoost：768原始数据
def ABC1():
    """AdaBoost:768原始数据"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 1.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["Pregnancies"] = label.fit_transform(dataset["Pregnancies"])
    dataset["Glucose"] = label.fit_transform(dataset["Glucose"])
    dataset["BloodPressure"] = label.fit_transform(dataset["BloodPressure"])
    dataset["SkinThickness"] = label.fit_transform(dataset["SkinThickness"])
    dataset["Insulin"] = label.fit_transform(dataset["Insulin"])
    dataset["BMI"] = label.fit_transform(dataset["BMI"])
    dataset["DiabetesPedigreeFunction"] = label.fit_transform(dataset["DiabetesPedigreeFunction"])
    dataset["Age"] = label.fit_transform(dataset["Age"])
    dataset["Outcome"] = label.fit_transform(dataset["Outcome"])

    # 定于预测目标变量
    target = ["Outcome"]

    # 定义模型的自变量
    train_x = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
               "Age"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用AdaBoost对数据集进行分类
    ABC1 = AdaBoostClassifier(
        random_state=16
    )
    ABC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    ABC1_train = ABC1.predict(X_train)
    ABC1_test = ABC1.predict(X_test)

    print("训练数据集上的精度:", accuracy_score(Y_train, ABC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, ABC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = ABC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("AdaBoost ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = ABC1.feature_importances_
    print(feat_important)


# AdaBoost：768关键特征
def ABC2():
    """AdaBoost:768关键特征"""
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

    # 使用AdaBoost对数据集进行分类
    ABC1 = AdaBoostClassifier(
        random_state=16
    )
    ABC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    ABC1_train = ABC1.predict(X_train)
    ABC1_test = ABC1.predict(X_test)

    print("训练数据集上的精度:", accuracy_score(Y_train, ABC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, ABC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = ABC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("AdaBoost ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = ABC1.feature_importances_
    print(feat_important)


# AdaBoost：2000原始数据
def ABC3():
    """AdaBoost:2000原始数据"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 4.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["Pregnancies00"] = label.fit_transform(dataset["Pregnancies00"])
    dataset["Glucose00"] = label.fit_transform(dataset["Glucose00"])
    dataset["BloodPressure00"] = label.fit_transform(dataset["BloodPressure00"])
    dataset["SkinThickness00"] = label.fit_transform(dataset["SkinThickness00"])
    dataset["Insulin00"] = label.fit_transform(dataset["Insulin00"])
    dataset["BMI00"] = label.fit_transform(dataset["BMI00"])
    dataset["DiabetesPedigreeFunction00"] = label.fit_transform(dataset["DiabetesPedigreeFunction00"])
    dataset["Age00"] = label.fit_transform(dataset["Age00"])
    dataset["Outcome00"] = label.fit_transform(dataset["Outcome00"])

    # 定于预测目标变量
    target = ["Outcome00"]

    # 定义模型的自变量
    train_x = ["Pregnancies00", "Glucose00", "BloodPressure00", "SkinThickness00",
               "Insulin00", "BMI00", "DiabetesPedigreeFunction00", "Age00"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用AdaBoost对数据集进行分类
    ABC1 = AdaBoostClassifier(
        random_state=16
    )
    ABC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    ABC1_train = ABC1.predict(X_train)
    ABC1_test = ABC1.predict(X_test)

    print("训练数据集上的精度:", accuracy_score(Y_train, ABC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, ABC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = ABC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("AdaBoost ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = ABC1.feature_importances_
    print(feat_important)


# AdaBoost：2000关键特征
def ABC4():
    """AdaBoost:2000关键特征"""
    # 导入数据，路径中要使用\\或者/或者在路径前面加r
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 6.csv')

    # 将字符串类型的分类变量重新编码
    label = LabelEncoder()
    dataset["GlucoseAge00"] = label.fit_transform(dataset["GlucoseAge00"])
    dataset["BMIDiabetesPedigreeFunction00"] = label.fit_transform(dataset["BMIDiabetesPedigreeFunction00"])
    dataset["AgeSkinThickness11"] = label.fit_transform(dataset["AgeSkinThickness11"])
    dataset["GlucoseGlucose11"] = label.fit_transform(dataset["GlucoseGlucose11"])
    dataset["GlucoseBloodPressure11"] = label.fit_transform(dataset["GlucoseBloodPressure11"])
    dataset["GlucoseBMI11"] = label.fit_transform(dataset["GlucoseBMI11"])
    dataset["BloodPressureBMI11"] = label.fit_transform(dataset["BloodPressureBMI11"])
    dataset["DiabetesPedigreeFunctionAge11"] = label.fit_transform(dataset["DiabetesPedigreeFunctionAge11"])
    dataset["Outcome33"] = label.fit_transform(dataset["Outcome33"])

    # 定于预测目标变量
    target = ["Outcome33"]

    # 定义模型的自变量
    train_x = ["GlucoseAge00", "BMIDiabetesPedigreeFunction00", "AgeSkinThickness11", "GlucoseGlucose11",
               "GlucoseBloodPressure11", "GlucoseBMI11", "BloodPressureBMI11", "DiabetesPedigreeFunctionAge11"]

    # 将数据集切分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    # 使用AdaBoost对数据集进行分类
    ABC1 = AdaBoostClassifier(
        random_state=16
    )
    ABC1.fit(X_train, Y_train)

    # 输出其在训练集和测试集上的预测精度
    ABC1_train = ABC1.predict(X_train)
    ABC1_test = ABC1.predict(X_test)

    print("训练数据集上的精度:", accuracy_score(Y_train, ABC1_train))
    print("验证数据集上的精度:", accuracy_score(Y_test, ABC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = ABC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  # 计算auc的取值
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("AdaBoost ROC曲线")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = ABC1.feature_importances_
    print(feat_important)



if __name__ == '__main__':

    RFC1()     # 随机森林 768原始数据     模型1
    # RFC2()     # 768最大相关最小冗余      模型2
    # RFC3()     # 768关键特征             模型3
    # RFC4()     # 2000原始数据            模型4
    # RFC5()     # 2000最大相关最小冗余     模型5
    # RFC6()     # 2000关键特征            模型6
    # RFC7()     # 768 因果置换            模型7
    # RFC8()     # 2000 因果置换           模型8

    # NNC1()     # 神经网络 768原始数据     模型9
    # NNC2()     # 768关键特征             模型10
    # NNC3()     # 2000原始数据            模型11
    # NNC4()     # 2000关键特征            模型12

    # IRC1()        # 逻辑回归 768原始数据         模型13
    # IRC2()        # 768关键特征                 模型14
    # IRC3()        # 2000原始数据                模型15
    # IRC4()        # 2000关键特征                模型16

    # ABC1()         # AdaBoost 768原始数据       模型17
    # ABC2()         # 768关键特征                模型18
    # ABC3()         # 2000原始数据               模型19
    # ABC4()         # 2000原始数据               模型20
