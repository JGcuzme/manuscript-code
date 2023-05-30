"""
Figure 14: ROC curves of feature set 6
"""

import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set(font="Kaiti", style="ticks", font_scale=1.4)
import pandas as pd
pd.set_option("max_colwidth", 200)
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

def o_auc_figure():
    dataset = pd.read_csv(r'..\data\feature set 6.csv')

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

    target = ["Outcome33"]

    train_x = ["GlucoseAge00", "BMIDiabetesPedigreeFunction00", "AgeSkinThickness11", "GlucoseGlucose11",
               "GlucoseBloodPressure11", "GlucoseBMI11", "BloodPressureBMI11", "DiabetesPedigreeFunctionAge11"]

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)

    RFC1 = RandomForestClassifier(
        oob_score=True,
        max_samples=978,
        random_state=123
    )
    RFC1.fit(X_train, Y_train)

    knn1 = KNeighborsClassifier(n_neighbors=12)
    knn1.fit(X_train, Y_train)

    IRC1 = LogisticRegression(
        random_state=123
    )
    IRC1.fit(X_train, Y_train)

    ABC1 = AdaBoostClassifier(
        random_state=123
    )
    ABC1.fit(X_train, Y_train)

    pre_test1 = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB1, TPR_NB1, _ = roc_curve(Y_test, pre_test1)
    aucval1 = auc(FPR_NB1, TPR_NB1)  

    pre_test2 = knn1.predict_proba(X_test)[:, 1]
    FPR_NB2, TPR_NB2, _ = roc_curve(Y_test, pre_test2)
    aucval2 = auc(FPR_NB2, TPR_NB2)  

    pre_test3 = IRC1.predict_proba(X_test)[:, 1]
    FPR_NB3, TPR_NB3, _ = roc_curve(Y_test, pre_test3)
    aucval3 = auc(FPR_NB3, TPR_NB3)  

    pre_test4 = ABC1.predict_proba(X_test)[:, 1]
    FPR_NB4, TPR_NB4, _ = roc_curve(Y_test, pre_test4)
    aucval4 = auc(FPR_NB4, TPR_NB4)  

    plt.figure(figsize=(8, 8), dpi=120)
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB1, TPR_NB1, "r", linewidth=3, label="Model 6(feature set 6 + RF)")
    plt.plot(FPR_NB2, TPR_NB2, "b", linewidth=3, label="Model 12(feature set 6 + KNN)")
    plt.plot(FPR_NB3, TPR_NB3, "g", linewidth=3, label="Model 16(feature set 6+ LR)")
    plt.plot(FPR_NB4, TPR_NB4, "black", linewidth=3, label="Model 20(feature set 6 + AdaBoost)")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("ROC curves of various algorithms for feature set 6")
    plt.text(0.65, 0.6, "AUC = " + str(round(aucval1, 2)), fontdict={"size": "20", "color": "r"})
    plt.text(0.65, 0.55, "AUC = " + str(round(aucval2, 2)), fontdict={"size": "20", "color": "b"})
    plt.text(0.65, 0.5, "AUC = " + str(round(aucval3, 2)), fontdict={"size": "20", "color": "g"})
    plt.text(0.65, 0.45, "AUC = " + str(("%.2f" % aucval4)), fontdict={"size": "20", "color": "black"})
    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':

    o_auc_figure()
