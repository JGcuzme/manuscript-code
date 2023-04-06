"""
Figure 13: ROC curves of feature set 4
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
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

def o_auc_figure():
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 4.csv')

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

    target = ["Outcome00"]

    train_x = ["Pregnancies00", "Glucose00", "BloodPressure00", "SkinThickness00",
               "Insulin00", "BMI00", "DiabetesPedigreeFunction00", "Age00"]

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  

    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=16
    )
    RFC1.fit(X_train, Y_train)

    MLP1 = MLPClassifier(
        random_state=16
    )
    MLP1.fit(X_train, Y_train)

    IRC1 = LogisticRegression(
        random_state=16
    )
    IRC1.fit(X_train, Y_train)

    ABC1 = AdaBoostClassifier(
        random_state=16
    )
    ABC1.fit(X_train, Y_train)

    pre_test1 = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB1, TPR_NB1, _ = roc_curve(Y_test, pre_test1)
    aucval1 = auc(FPR_NB1, TPR_NB1)  

    pre_test2 = MLP1.predict_proba(X_test)[:, 1]
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
    plt.plot(FPR_NB1, TPR_NB1, "r", linewidth=3, label="Model 4(feature set 4 + RF)")
    plt.plot(FPR_NB2, TPR_NB2, "b", linewidth=3, label="Model 11(feature set 4 + ANN)")
    plt.plot(FPR_NB3, TPR_NB3, "g", linewidth=3, label="Model 15(feature set 4+ LR)")
    plt.plot(FPR_NB4, TPR_NB4, "black", linewidth=3, label="Model 19(feature set 4 + AdaBoost)")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("ROC curves of various algorithms for feature set 4")
    plt.text(0.65, 0.6, "AUC = " + str(round(aucval1, 2)), fontdict={"size": "20", "color": "r"})
    plt.text(0.65, 0.55, "AUC = " + str(round(aucval2, 2)), fontdict={"size": "20", "color": "b"})
    plt.text(0.65, 0.5, "AUC = " + str(round(aucval3, 2)), fontdict={"size": "20", "color": "g"})
    plt.text(0.65, 0.45, "AUC = " + str(round(aucval4, 2)), fontdict={"size": "20", "color": "black"})
    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':

    o_auc_figure()