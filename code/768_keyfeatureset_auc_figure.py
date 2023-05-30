"""
Figure 12: ROC curves of feature set 3
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

def k_auc_figure():
    
    dataset = pd.read_csv(r'..\data\feature set 3.csv')

    
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


    target = ["Outcome3"]


    train_x = ["GlucoseBMI", "AgeAge1", "AgeDiabetesPedigreeFunction1", "GlucoseGlucose1",
               "GlucoseBloodPressure1", "BloodPressureBMI1", "BMIAge1", "BMIBMI1"]


    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)

    RFC1 = RandomForestClassifier(
        oob_score=True,
        max_samples=46,
        random_state=123
    )
    RFC1.fit(X_train, Y_train)

    knn1 = KNeighborsClassifier(n_neighbors=7)
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
    plt.plot(FPR_NB1, TPR_NB1, "r", linewidth=3, label="Model 3(feature set 3 + RF)")
    plt.plot(FPR_NB2, TPR_NB2, "b", linewidth=3, label="Model 10(feature set 3 + KNN)")
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
