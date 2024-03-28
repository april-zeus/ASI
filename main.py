import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import warnings

# Suppressing warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read the data
df = pd.read_csv("data.csv", index_col=0)

# Categorical Variables
categorical_variables = [col for col in df.columns if col in "O"
                         or df[col].nunique() <= 11
                         and col not in "Exited"]

# Numeric Variables
numeric_variables = [col for col in df.columns if df[col].dtype != "object"
                     and df[col].nunique() > 11
                     and col not in "CustomerId"]

# Separating churned and not churned customers
churn = df.loc[df["Exited"] == 1]
not_churn = df.loc[df["Exited"] == 0]

# # To determine the threshold value for outliers
# def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
#     quantile_one = dataframe[variable].quantile(low_quantile)
#     quantile_three = dataframe[variable].quantile(up_quantile)
#     interquantile_range = quantile_three - quantile_one
#     up_limit = quantile_three + 1.5 * interquantile_range
#     low_limit = quantile_one - 1.5 * interquantile_range
#     return low_limit, up_limit
#
# # Are there any outliers in the variables
# def has_outliers(dataframe, numeric_columns, plot=False):
#     # variable_names = []
#     for col in numeric_columns:
#         low_limit, up_limit = outlier_thresholds(dataframe, col)
#         if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
#             number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
#             print(col, " : ", number_of_outliers, "outliers")
#             # variable_names.append(col)
#             if plot:
#                 sns.boxplot(x=dataframe[col])
#                 plt.show()
#     # return variable_names
#
# # There is no outlier
# for var in numeric_variables:
#     print(var, "has ", has_outliers(df, [var]), "Outliers")


# Feature engineering
df["NewTenure"] = df["Tenure"] / df["Age"]
df["NewCreditsScore"] = pd.qcut(df['CreditScore'], 6, labels=[1, 2, 3, 4, 5, 6])
df["NewAgeScore"] = pd.qcut(df['Age'], 8, labels=[1, 2, 3, 4, 5, 6, 7, 8])
df["NewBalanceScore"] = pd.qcut(df['Balance'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
df["NewEstSalaryScore"] = pd.qcut(df['EstimatedSalary'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# One hot encoding
columns_to_encode = ["Gender", "Geography"]
df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

# Dropping unnecessary variables
df = df.drop(["CustomerId", "Surname"], axis=1)


# Robust scaling
def robust_scaler(variable):
    var_median = variable.median()
    quartile1 = variable.quantile(0.25)
    quartile3 = variable.quantile(0.75)
    interquantile_range = quartile3 - quartile1
    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.05)
        quartile3 = variable.quantile(0.95)
        interquantile_range = quartile3 - quartile1
        if int(interquantile_range) == 0:
            quartile1 = variable.quantile(0.01)
            quartile3 = variable.quantile(0.99)
            interquantile_range = quartile3 - quartile1
            z = (variable - var_median) / interquantile_range
            return round(z, 3)

        z = (variable - var_median) / interquantile_range
        return round(z, 3)
    else:
        z = (variable - var_median) / interquantile_range
    return round(z, 3)


like_num = [col for col in df.columns if df[col].dtypes != 'O' and len(df[col].value_counts()) <= 10]
columns_to_scale = [col for col in df.columns if col not in ["Gender_Male", "Geography_Germany", "Geography_Spain"]
                    and col not in "Exited"
                    and col not in like_num]

for col in columns_to_scale:
    df[col] = robust_scaler(df[col])

X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12345)

# # Models for Classification
# models = [('LR', LogisticRegression(random_state=123456)),
#           ('KNN', KNeighborsClassifier()),
#           ('CART', DecisionTreeClassifier(random_state=123456)),
#           ('RF', RandomForestClassifier(random_state=123456)),
#           ('SVR', SVC(gamma='auto', random_state=123456)),
#           ('GB', GradientBoostingClassifier(random_state=12345)),
#           ("LightGBM", LGBMClassifier(random_state=123456))]
# results = []
# names = []
# for name, model in models:
#     kfold = KFold(n_splits=10, random_state=123456, shuffle=True)
#     cv_results = cross_val_score(model, X, y, cv=kfold)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)


# Gradient Boosting Classifier
model_GB = GradientBoostingClassifier(random_state=12345)
model_GB.fit(X_train, y_train)
y_pred = model_GB.predict(X_test)
conf_mat = confusion_matrix(y_pred, y_test)

# print("True Positive : ", conf_mat[1, 1])
# print("True Negative : ", conf_mat[0, 0])
# print("False Positive: ", conf_mat[0, 1])
# print("False Negative: ", conf_mat[1, 0])

# Classification Report for Gradient Boosting Model
print(classification_report(model_GB.predict(X_test), y_test))


# Plotting AUC ROC Curve
def generate_auc_roc_curve(clf, x_test):
    y_pred_proba = clf.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="AUC ROC Curve with Area Under the curve =" + str(auc))
    plt.legend(loc=4)
    plt.show()


generate_auc_roc_curve(model_GB, X_test)

# LightGBM Classifier
lgb_model = LGBMClassifier()

# LightGBM Model Tuning
lgbm_params = {'colsample_bytree': 0.5,
               'learning_rate': 0.01,
               'max_depth': 6,
               'n_estimators': 500}

lgbm_tuned = LGBMClassifier(**lgbm_params).fit(X, y)

# Gradient Boosting Model Tuning
gbm_model = GradientBoostingClassifier()
gbm_params = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1}
gbm_tuned = GradientBoostingClassifier(**gbm_params).fit(X, y)

# Model Evaluation
models = [("LightGBM", lgbm_tuned), ("GB", gbm_tuned)]
for name, model in models:
    kfold = KFold(n_splits=10, random_state=123456, shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    print(f"{name}: Mean Accuracy: {cv_results.mean()}, Standard Deviation: {cv_results.std()}")

# Feature Importance Visualization
for name, model in models:
    base = model.fit(X_train, y_train)
    y_pred = base.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    feature_imp = pd.Series(base.feature_importances_, index=X.columns).sort_values(ascending=False)

    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Scores')
    plt.ylabel('Features')
    plt.title(name)
    plt.show()
