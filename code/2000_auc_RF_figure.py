"""
Figure 4: ROC curve with sample size of 2000
"""

import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set(font="Kaiti", style="ticks", font_scale=1.4)
import pandas as pd
pd.set_option("max_colwidth", 200)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import warnings
warnings.filterwarnings("ignore")

dataset1 = pd.read_csv(r'..\data\feature set 4.csv')

label = LabelEncoder()
dataset1["Pregnancies00"] = label.fit_transform(dataset1["Pregnancies00"])
dataset1["Glucose00"] = label.fit_transform(dataset1["Glucose00"])
dataset1["BloodPressure00"] = label.fit_transform(dataset1["BloodPressure00"])
dataset1["SkinThickness00"] = label.fit_transform(dataset1["SkinThickness00"])
dataset1["Insulin00"] = label.fit_transform(dataset1["Insulin00"])
dataset1["BMI00"] = label.fit_transform(dataset1["BMI00"])
dataset1["DiabetesPedigreeFunction00"] = label.fit_transform(dataset1["DiabetesPedigreeFunction00"])
dataset1["Age00"] = label.fit_transform(dataset1["Age00"])
dataset1["Outcome00"] = label.fit_transform(dataset1["Outcome00"])

target1 = ["Outcome00"]

train_x1 = ["Pregnancies00", "Glucose00", "BloodPressure00", "SkinThickness00",
           "Insulin00", "BMI00", "DiabetesPedigreeFunction00", "Age00"]

X1_train, X1_test, Y1_train, Y1_test = train_test_split(dataset1[train_x1], dataset1[target1],
                                                 test_size=0.3, random_state=123)
RFC1 = RandomForestClassifier(
                                oob_score=True,
                                max_samples=978,
                                random_state=123
)
RFC1.fit(X1_train, Y1_train)

dataset2 = pd.read_csv(r'..\data\feature set 5.csv')

label = LabelEncoder()
dataset2["GlucoseAge00"] = label.fit_transform(dataset2["GlucoseAge00"])
dataset2["BMIDiabetesPedigreeFunction00"] = label.fit_transform(dataset2["BMIDiabetesPedigreeFunction00"])
dataset2["PregnanciesSkinThickness11"] = label.fit_transform(dataset2["PregnanciesSkinThickness11"])
dataset2["GlucoseGlucose11"] = label.fit_transform(dataset2["GlucoseGlucose11"])
dataset2["GlucoseBloodPressure11"] = label.fit_transform(dataset2["GlucoseBloodPressure11"])
dataset2["GlucoseBMI11"] = label.fit_transform(dataset2["GlucoseBMI11"])
dataset2["BloodPressureBMI11"] = label.fit_transform(dataset2["BloodPressureBMI11"])
dataset2["DiabetesPedigreeFunctionAge11"] = label.fit_transform(dataset2["DiabetesPedigreeFunctionAge11"])
dataset2["Outcome11"] = label.fit_transform(dataset2["Outcome11"])

target2 = ["Outcome11"]

train_x2 = ["GlucoseAge00","BMIDiabetesPedigreeFunction00","PregnanciesSkinThickness11","GlucoseGlucose11",
           "GlucoseBloodPressure11","GlucoseBMI11","BloodPressureBMI11","DiabetesPedigreeFunctionAge11"]

X2_train, X2_test, Y2_train, Y2_test = train_test_split(dataset2[train_x2], dataset2[target2],
                                                 test_size=0.3, random_state=123)

RFC2 = RandomForestClassifier(
                                oob_score=True,
                                max_samples=978,
                                random_state=123
)
RFC2.fit(X2_train, Y2_train)

dataset3 = pd.read_csv(r'..\data\feature set 6.csv')

label = LabelEncoder()
dataset3["GlucoseAge00"] = label.fit_transform(dataset3["GlucoseAge00"])
dataset3["BMIDiabetesPedigreeFunction00"] = label.fit_transform(dataset3["BMIDiabetesPedigreeFunction00"])
dataset3["AgeSkinThickness11"] = label.fit_transform(dataset3["AgeSkinThickness11"])
dataset3["GlucoseGlucose11"] = label.fit_transform(dataset3["GlucoseGlucose11"])
dataset3["GlucoseBloodPressure11"] = label.fit_transform(dataset3["GlucoseBloodPressure11"])
dataset3["GlucoseBMI11"] = label.fit_transform(dataset3["GlucoseBMI11"])
dataset3["BloodPressureBMI11"] = label.fit_transform(dataset3["BloodPressureBMI11"])
dataset3["DiabetesPedigreeFunctionAge11"] = label.fit_transform(dataset3["DiabetesPedigreeFunctionAge11"])
dataset3["Outcome33"] = label.fit_transform(dataset3["Outcome33"])

target3 = ["Outcome33"]

train_x3 = ["GlucoseAge00", "BMIDiabetesPedigreeFunction00", "AgeSkinThickness11", "GlucoseGlucose11",
           "GlucoseBloodPressure11", "GlucoseBMI11", "BloodPressureBMI11", "DiabetesPedigreeFunctionAge11"]

X3_train, X3_test, Y3_train, Y3_test = train_test_split(dataset3[train_x3], dataset3[target3],
                                                 test_size=0.3, random_state=123)

RFC3 = RandomForestClassifier(
                                oob_score=True,
                                max_samples=978,
                                random_state=123
)
RFC3.fit(X3_train, Y3_train)

pre_test1 = RFC1.predict_proba(X1_test)[:, 1]
FPR_NB1, TPR_NB1, _ = roc_curve(Y1_test, pre_test1)
aucval1 = auc(FPR_NB1, TPR_NB1)

pre_test2 = RFC2.predict_proba(X2_test)[:, 1]
FPR_NB2, TPR_NB2, _ = roc_curve(Y2_test, pre_test2)
aucval2 = auc(FPR_NB2, TPR_NB2)

pre_test3 = RFC3.predict_proba(X3_test)[:, 1]
FPR_NB3, TPR_NB3, _ = roc_curve(Y3_test, pre_test3)
aucval3 = auc(FPR_NB3, TPR_NB3)

plt.figure(figsize=(8, 8), dpi=120)
plt.plot([0, 1], [0, 1], "k--")
plt.plot(FPR_NB1, TPR_NB1, "r", linewidth=3, label="Model 4(feature set 4 + RF)")
plt.plot(FPR_NB2, TPR_NB2, "b", linewidth=3, label="Model 5(feature set 5 + RF)")
plt.plot(FPR_NB3, TPR_NB3, "g", linewidth=3, label="Model 6(feature set 6 + RF)")
plt.grid()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Random forest ROC curve of 2000 samples")
plt.text(0.05, 0.9, "AUC = "+str(round(aucval1, 2)), fontdict={"size": "20", "color": "r"})
plt.text(0.05, 0.85, "AUC = "+str(round(aucval2, 2)), fontdict={"size": "20", "color": "b"})
plt.text(0.05, 0.8, "AUC = "+str(round(aucval3, 2)), fontdict={"size": "20", "color": "g"})
plt.legend(loc=4)
plt.show()
