"""
Figure 11: ROC curves of feature set 1
"""


import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set(font="Kaiti", style="ticks", font_scale=1.4)
import pandas as pd"""
Figure 11: ROC curves of feature set 1
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

def o_auc_figure():
    
    dataset = pd.read_csv(r'..\data\feature set 1.csv')

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

    target = ["Outcome"]

    train_x = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
               "Age"]

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  

    RFC1 = RandomForestClassifier(
        oob_score=True,
        max_samples=46,
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
    plt.plot(FPR_NB1, TPR_NB1, "r", linewidth=3, label="Model 1(feature set 1 + RF)")
    plt.plot(FPR_NB2, TPR_NB2, "b", linewidth=3, label="Model 9(feature set 1 + KNN)")
    plt.plot(FPR_NB3, TPR_NB3, "g", linewidth=3, label="Model 13(feature set 1 + LR)")
    plt.plot(FPR_NB4, TPR_NB4, "black", linewidth=3, label="Model 17(feature set 1 + AdaBoost)")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("ROC curves of various algorithms for feature set 1")
    plt.text(0.65, 0.6, "AUC = " + str(round(aucval1, 2)), fontdict={"size": "20", "color": "r"})
    plt.text(0.65, 0.55, "AUC = " + str(round(aucval2, 2)), fontdict={"size": "20", "color": "b"})
    plt.text(0.65, 0.5, "AUC = " + str(round(aucval3, 2)), fontdict={"size": "20", "color": "g"})
    plt.text(0.65, 0.45, "AUC = " + str(round(aucval4, 2)), fontdict={"size": "20", "color": "black"})
    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':

    o_auc_figure()

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
    
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 1.csv')

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

    target = ["Outcome"]

    train_x = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
               "Age"]

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  

    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=16
    )
    RFC1.fit(X_train, Y_train)

    MLP1 = MLPClassifier(
        random_state=63
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
    plt.plot(FPR_NB1, TPR_NB1, "r", linewidth=3, label="Model 1(feature set 1 + RF)")
    plt.plot(FPR_NB2, TPR_NB2, "b", linewidth=3, label="Model 9(feature set 1 + ANN)")
    plt.plot(FPR_NB3, TPR_NB3, "g", linewidth=3, label="Model 13(feature set 1 + LR)")
    plt.plot(FPR_NB4, TPR_NB4, "black", linewidth=3, label="Model 17(feature set 1 + AdaBoost)")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("ROC curves of various algorithms for feature set 1")
    plt.text(0.65, 0.6, "AUC = " + str(round(aucval1, 2)), fontdict={"size": "20", "color": "r"})
    plt.text(0.65, 0.55, "AUC = " + str(round(aucval2, 2)), fontdict={"size": "20", "color": "b"})
    plt.text(0.65, 0.5, "AUC = " + str(round(aucval3, 2)), fontdict={"size": "20", "color": "g"})
    plt.text(0.65, 0.45, "AUC = " + str(round(aucval4, 2)), fontdict={"size": "20", "color": "black"})
    plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':

    o_auc_figure()
