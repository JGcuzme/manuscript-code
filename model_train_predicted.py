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
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import *



def RFC1():
    """Random Forest: 768 original data"""

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

    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, RFC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, RFC1_test))

    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Random forest ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)


def RFC2():
    """Random Forest: 768 Important Feature sets"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 2.csv')


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

    target = ["Outcome1"]

    train_x = ["GlucoseBMI", "PregnanciesDiabetesPedigreeFunction1", "PregnanciesAge1", "GlucoseGlucose1",
               "GlucoseBloodPressure1", "BloodPressureBMI1", "SkinThicknessBMI1", "SkinThicknessAge1"]

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  

    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=16
    )
    RFC1.fit(X_train, Y_train)

    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, RFC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, RFC1_test))

    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Random forest ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)


def RFC3():
    """Random Forest: 768 Key Features"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 3.csv')

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
                                    random_state=16
                                )
    RFC1.fit(X_train, Y_train)

    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, RFC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, RFC1_test))

    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Random forest ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)


def RFC4():
    """Random Forest: 2000 original data"""
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
        random_state=22)
    RFC1.fit(X_train, Y_train)

    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, RFC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, RFC1_test))

    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Random forest ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)


def RFC5():
    """Random Forest: 2000 Important Feature sets"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 5.csv')

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

    target = ["Outcome11"]

    train_x = ["GlucoseAge00", "BMIDiabetesPedigreeFunction00", "PregnanciesSkinThickness11", "GlucoseGlucose11",
               "GlucoseBloodPressure11", "GlucoseBMI11", "BloodPressureBMI11", "DiabetesPedigreeFunctionAge11"]

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=22)
    RFC1.fit(X_train, Y_train)

    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, RFC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, RFC1_test))

    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Random forest ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)


def RFC6():
    """Random Forest: 2000 Key Features"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 6.csv')

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
                                                        test_size=0.3, random_state=123)  # random_state 是随机数种子

    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=22)
    RFC1.fit(X_train, Y_train)

    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, RFC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, RFC1_test))

    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Random forest ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()
    feat_important = RFC1.feature_importances_
    print(feat_important)


def RFC7():
    """Random Forest: 768 Causal Permutation"""
    dataset = pd.read_csv(r'C:.\Supplemental Files\data\feature set 7.csv')

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

    target = ["Outcome5"]

    train_x = ["GlucoseSkinThickness", "PregnanciesDiabetesPedigreeFunction1", "PregnanciesPregnancies1",
               "GlucoseGlucose1", "GlucoseBloodPressure1", "BloodPressureSkinThickness1",
               "SkinThicknessSkinThickness1", "SkinThicknessPregnancies1"]

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  

    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=16)
    RFC1.fit(X_train, Y_train)

    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, RFC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, RFC1_test))

    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Random forest ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()
    feat_important = RFC1.feature_importances_
    print(feat_important)


def RFC8():
    """Random Forest: 2000 Causal Permutation"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 8.csv')

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

    target = ["Outcome55"]

    train_x = ["PregnanciesGlucose00", "SkinThicknessDiabetesPedigreeFunction00", "PregnanciesSkinThickness11",
               "GlucoseGlucose11", "GlucoseBloodPressure11", "GlucoseSkinThickness11",
               "BloodPressureSkinThickness11", "DiabetesPedigreeFunctionPregnancies11"]

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=123)  

    RFC1 = RandomForestClassifier(
        oob_score=True,
        random_state=22)
    RFC1.fit(X_train, Y_train)

    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, RFC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, RFC1_test))

    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Random forest ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()
    feat_important = RFC1.feature_importances_
    print(feat_important)


def NNC1():
    """Neural networks: 768 original data"""
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 1.csv')

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

    MLP1 = MLPClassifier(
                            random_state=63
                        )
    MLP1.fit(X_train, Y_train)

    MLP1_train = MLP1.predict(X_train)
    MLP1_test = MLP1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, MLP1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, MLP1_test))

    pre_test = MLP1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Neural network ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


def NNC2():
    """Neural networks: 768 Key features"""
    dataset = pd.read_csv(r'C:\users\14903\Desktop\Supplemental Files\data\feature set 3.csv')

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

    MLP1 = MLPClassifier(
        random_state=63
    )
    MLP1.fit(X_train, Y_train)

    MLP1_train = MLP1.predict(X_train)
    MLP1_test = MLP1.predict(X_test)
    print("Precision on the training data set:", accuracy_score(Y_train, MLP1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, MLP1_test))

    pre_test = MLP1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Neural network ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


def NNC3():
    """Neural networks: 2000 original data"""

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

    MLP1 = MLPClassifier(
        random_state=16
    )
    MLP1.fit(X_train, Y_train)

    MLP1_train = MLP1.predict(X_train)
    MLP1_test = MLP1.predict(X_test)
    print("Precision on the training data set:", accuracy_score(Y_train, MLP1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, MLP1_test))

    pre_test = MLP1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Neural network ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


def NNC4():
    """Neural networks: 2000 Key features"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 6.csv')

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

    MLP1 = MLPClassifier(
        random_state=16
    )
    MLP1.fit(X_train, Y_train)

    MLP1_train = MLP1.predict(X_train)
    MLP1_test = MLP1.predict(X_test)
    print("Precision on the training data set:", accuracy_score(Y_train, MLP1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, MLP1_test))

    pre_test = MLP1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Neural network ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()



def IRC1():
    """Logistic regression: 768 original data"""
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

    IRC1 = LogisticRegression(random_state=16)
    IRC1.fit(X_train, Y_train)

    IRC1_train = IRC1.predict(X_train)
    IRC1_test = IRC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, IRC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, IRC1_test))

    # 可视化在测试集上的ROC曲线
    pre_test = IRC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Logistic regression ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


def IRC2():
    """Logistic Regression: 768 Key features"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 3.csv')

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

    IRC1 = LogisticRegression(random_state=16)
    IRC1.fit(X_train, Y_train)

    IRC1_train = IRC1.predict(X_train)
    IRC1_test = IRC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, IRC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, IRC1_test))

    pre_test = IRC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Logistic regression ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


def IRC3():
    """Logistic regression: 2000 original data"""
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

    IRC1 = LogisticRegression(random_state=16)
    IRC1.fit(X_train, Y_train)

    IRC1_train = IRC1.predict(X_train)
    IRC1_test = IRC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, IRC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, IRC1_test))

    pre_test = IRC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Logistic regression ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


def IRC4():
    """Logistic Regression: 2000 Key features"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 6.csv')

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

    IRC1 = LogisticRegression(random_state=16)
    IRC1.fit(X_train, Y_train)

    IRC1_train = IRC1.predict(X_train)
    IRC1_test = IRC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, IRC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, IRC1_test))

    pre_test = IRC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Logistic regression ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()


def ABC1():
    """AdaBoost:768 Original data"""

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

    ABC1 = AdaBoostClassifier(
        random_state=16
    )
    ABC1.fit(X_train, Y_train)

    ABC1_train = ABC1.predict(X_train)
    ABC1_test = ABC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, ABC1_train))
    print("Precision on the testig data set:", accuracy_score(Y_test, ABC1_test))

    pre_test = ABC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("AdaBoost ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = ABC1.feature_importances_
    print(feat_important)


def ABC2():
    """AdaBoost:768 key features"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 3.csv')

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

    ABC1 = AdaBoostClassifier(
        random_state=16
    )
    ABC1.fit(X_train, Y_train)

    ABC1_train = ABC1.predict(X_train)
    ABC1_test = ABC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, ABC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, ABC1_test))

    pre_test = ABC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("AdaBoost ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = ABC1.feature_importances_
    print(feat_important)


def ABC3():
    """AdaBoost:2000 Original data"""
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

    ABC1 = AdaBoostClassifier(
        random_state=16
    )
    ABC1.fit(X_train, Y_train)

    ABC1_train = ABC1.predict(X_train)
    ABC1_test = ABC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, ABC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, ABC1_test))

    pre_test = ABC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("AdaBoost ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = ABC1.feature_importances_
    print(feat_important)


def ABC4():
    """AdaBoost:2000 Key features"""

    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 6.csv')

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

    ABC1 = AdaBoostClassifier(
        random_state=16
    )
    ABC1.fit(X_train, Y_train)

    ABC1_train = ABC1.predict(X_train)
    ABC1_test = ABC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, ABC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, ABC1_test))

    pre_test = ABC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("AdaBoost ROC curve")
    plt.text(0.05, 0.9, "AUC = " + str(round(aucval, 2)))
    plt.show()

    feat_important = ABC1.feature_importances_
    print(feat_important)

    
def NHANES1():
    """Random Forest: NHANES data"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 9.csv')

    label = LabelEncoder()
    dataset["RIAGENDR"] = label.fit_transform(dataset["RIAGENDR"])   
    dataset["RIDAGEYR"] = label.fit_transform(dataset["RIDAGEYR"])   
    dataset["RIDRETH3"] = label.fit_transform(dataset["RIDRETH3"])   
    dataset["BPXSY1"] = label.fit_transform(dataset["BPXSY1"])       
    dataset["BPXDI1"] = label.fit_transform(dataset["BPXDI1"])       
    dataset["BMXWT"] = label.fit_transform(dataset["BMXWT"])        
    dataset["BMXBMI"] = label.fit_transform(dataset["BMXBMI"])     
    dataset["URXUMA"] = label.fit_transform(dataset["URXUMA"])      
    dataset["URXUCR"] = label.fit_transform(dataset["URXUCR"])       
    dataset["URDACT"] = label.fit_transform(dataset["URDACT"])       
    dataset["LBDHDD"] = label.fit_transform(dataset["LBDHDD"])       
    dataset["LBXTR"] = label.fit_transform(dataset["LBXTR"])         
    dataset["LBDLDL"] = label.fit_transform(dataset["LBDLDL"])       
    dataset["LBXTC"] = label.fit_transform(dataset["LBXTC"])         
    dataset["LBXGH"] = label.fit_transform(dataset["LBXGH"])         
    dataset["LBXIN"] = label.fit_transform(dataset["LBXIN"])         
    dataset["LBXGLU"] = label.fit_transform(dataset["LBXGLU"])       
    dataset["Outcome"] = label.fit_transform(dataset["DIQ010"])

    target = ["Outcome"]

    train_x = ["RIDAGEYR", "RIDRETH3", "BPXSY1", "BPXDI1",
               "BMXWT", "BMXBMI", "URXUMA", "URXUCR", "URDACT",
               "LBDHDD", "LBXTR", "LBDLDL", "LBXTC", "LBXGH",
               "LBXIN", "LBXGLU"]

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=215)  


    RFC1 = RandomForestClassifier(
                                  oob_score=True,
                                  random_state=215)
    RFC1.fit(X_train, Y_train)

    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, RFC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, RFC1_test))

    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Random forest ROC curve")
    plt.text(0.05, 0.9, "AUC = "+str(round(aucval, 4)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)
    
 def NHANES2():
    """Random Forest: NHANES important features"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 10.csv')

    label = LabelEncoder()
    dataset["RIAGENDR 2"] = label.fit_transform(dataset["RIAGENDR 2"])
    dataset["BMXWTLBXGH"] = label.fit_transform(dataset["BMXWTLBXGH"])
    dataset["RIAGENDR 1"] = label.fit_transform(dataset["RIAGENDR 1"])
    dataset["RIAGENDR 3"] = label.fit_transform(dataset["RIAGENDR 3"])
    dataset["RIAGENDRRIAGENDR 1"] = label.fit_transform(dataset["RIAGENDRRIAGENDR 1"])
    dataset["BPXSY1RIAGENDR 1"] = label.fit_transform(dataset["BPXSY1RIAGENDR 1"])
    dataset["LBXGHRIAGENDR 1"] = label.fit_transform(dataset["LBXGHRIAGENDR 1"])
    dataset["RIAGENDRRIDAGEYR 1"] = label.fit_transform(dataset["RIAGENDRRIDAGEYR 1"])
    dataset["URXUCRLBXGLU"] = label.fit_transform(dataset["URXUCRLBXGLU"])
    dataset["RIAGENDRBPXSY1"] = label.fit_transform(dataset["RIAGENDRBPXSY1"])
    dataset["RIAGENDRLBXGH"] = label.fit_transform(dataset["RIAGENDRLBXGH"])
    dataset["RIDRETH3 3"] = label.fit_transform(dataset["RIDRETH3 3"])
    dataset["LBXGLURIAGENDR 1"] = label.fit_transform(dataset["LBXGLURIAGENDR 1"])
    dataset["LBXTR 1"] = label.fit_transform(dataset["LBXTR 1"])
    dataset["URXUCRRIDAGEYR 1"] = label.fit_transform(dataset["URXUCRRIDAGEYR 1"])
    dataset["LBXGLURIDRETH3 1"] = label.fit_transform(dataset["LBXGLURIDRETH3 1"])
    dataset["LBXTRBMXWT 1"] = label.fit_transform(dataset["LBXTRBMXWT 1"])
    dataset["Outcome"] = label.fit_transform(dataset["DIQ010"])

    target = ["Outcome"]

    train_x = ["RIAGENDR 2", "BMXWTLBXGH", "RIAGENDR 1", "RIAGENDR 3", "RIAGENDRRIAGENDR 1",
               "BPXSY1RIAGENDR 1", "LBXGHRIAGENDR 1", "RIAGENDRRIDAGEYR 1", "URXUCRLBXGLU", "RIAGENDRBPXSY1",
               "RIAGENDRLBXGH", "RIDRETH3 3", "LBXGLURIAGENDR 1", "LBXTR 1", "URXUCRRIDAGEYR 1",
               "LBXGLURIDRETH3 1", "LBXTRBMXWT 1"]

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=215) 


    RFC1 = RandomForestClassifier(
                                  oob_score=True,
                                  random_state=215)
    RFC1.fit(X_train, Y_train)

    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, RFC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, RFC1_test))

    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB)  
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Random forest ROC curve")
    plt.text(0.05, 0.9, "AUC = "+str(round(aucval, 4)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)
    
def NHANES3():
    """Random Forest: NHANES Key features"""
    dataset = pd.read_csv(r'.\Supplemental Files\data\feature set 11.csv')

    label = LabelEncoder()

    dataset["RIAGENDR 2"] = label.fit_transform(dataset["RIAGENDR 2"])
    dataset["BMXWTLBXGH"] = label.fit_transform(dataset["BMXWTLBXGH"])
    dataset["RIAGENDR 1"] = label.fit_transform(dataset["RIAGENDR 1"])
    dataset["RIAGENDR 3"] = label.fit_transform(dataset["RIAGENDR 3"])
    dataset["RIAGENDRRIAGENDR 1"] = label.fit_transform(dataset["RIAGENDRRIAGENDR 1"])
    dataset["BPXSY1RIAGENDR 1"] = label.fit_transform(dataset["BPXSY1RIAGENDR 1"])
    dataset["LBXGHRIAGENDR 1"] = label.fit_transform(dataset["LBXGHRIAGENDR 1"])
    dataset["RIAGENDRRIDAGEYR 1"] = label.fit_transform(dataset["RIAGENDRRIDAGEYR 1"])
    dataset["URXUCRLBXGLU"] = label.fit_transform(dataset["URXUCRLBXGLU"])
    dataset["RIAGENDRBPXSY1"] = label.fit_transform(dataset["RIAGENDRBPXSY1"])
    dataset["RIAGENDRLBXGH"] = label.fit_transform(dataset["RIAGENDRLBXGH"])
    dataset["RIDRETH3 3"] = label.fit_transform(dataset["RIDRETH3 3"])
    dataset["LBXGLURIAGENDR 1"] = label.fit_transform(dataset["LBXGLURIAGENDR 1"])
    dataset["LBDLDL 1"] = label.fit_transform(dataset["LBDLDL 1"])
    dataset["URXUCRRIDAGEYR 1"] = label.fit_transform(dataset["URXUCRRIDAGEYR 1"])
    dataset["LBXGLURIDRETH3 1"] = label.fit_transform(dataset["LBXGLURIDRETH3 1"])
    dataset["LBDLDLBMXWT 1"] = label.fit_transform(dataset["LBDLDLBMXWT 1"])
    dataset["Outcome"] = label.fit_transform(dataset["DIQ010"])

    target = ["Outcome"]

    train_x = ["RIAGENDR 2", "BMXWTLBXGH", "RIAGENDR 1", "RIAGENDR 3", "RIAGENDRRIAGENDR 1",
               "BPXSY1RIAGENDR 1", "LBXGHRIAGENDR 1", "RIAGENDRRIDAGEYR 1", "URXUCRLBXGLU", "RIAGENDRBPXSY1",
               "RIAGENDRLBXGH", "RIDRETH3 3", "LBXGLURIAGENDR 1", "LBDLDL 1", "URXUCRRIDAGEYR 1",
               "LBXGLURIDRETH3 1", "LBDLDLBMXWT 1"]

    X_train, X_test, Y_train, Y_test = train_test_split(dataset[train_x], dataset[target],
                                                        test_size=0.3, random_state=215) 


    RFC1 = RandomForestClassifier(
                                  oob_score=True,
                                  random_state=215)
    RFC1.fit(X_train, Y_train)

    RFC1_train = RFC1.predict(X_train)
    RFC1_test = RFC1.predict(X_test)

    print("Precision on the training data set:", accuracy_score(Y_train, RFC1_train))
    print("Precision on the testing data set:", accuracy_score(Y_test, RFC1_test))

    pre_test = RFC1.predict_proba(X_test)[:, 1]
    FPR_NB, TPR_NB, _ = roc_curve(Y_test, pre_test)
    aucval = auc(FPR_NB, TPR_NB) 
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(FPR_NB, TPR_NB, "r", linewidth=3)
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Random forest ROC curve")
    plt.text(0.05, 0.9, "AUC = "+str(round(aucval, 4)))
    plt.show()

    feat_important = RFC1.feature_importances_
    print(feat_important)

if __name__ == '__main__':

    RFC1()     # model 1
    # RFC2()     # model 2
    # RFC3()     # model 3
    # RFC4()     # model 4
    # RFC5()     # model 5
    # RFC6()     # model 6
    # RFC7()     # model 7
    # RFC8()     # model 8

    # NNC1()     # model 9
    # NNC2()     # model 10
    # NNC3()     # model 11
    # NNC4()     # model 12

    # IRC1()        # model 13
    # IRC2()        # model 14
    # IRC3()        # model 15
    # IRC4()        # model 16

    # ABC1()         # model 17
    # ABC2()         # model 18
    # ABC3()         # model 19
    # ABC4()         # model 20
    
    # NHANES1()         # model 21
    # NHANES2()         # model 22
    # NHANES3()         # model 23
