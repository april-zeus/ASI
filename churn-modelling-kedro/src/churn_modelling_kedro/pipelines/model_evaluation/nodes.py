import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(predictions: pd.DataFrame, test_labels: pd.DataFrame):
    # conf_mat = confusion_matrix(predictions, test_labels)
    #
    # print("True Positive : ", conf_mat[1, 1])
    # print("True Negative : ", conf_mat[0, 0])
    # print("False Positive: ", conf_mat[0, 1])
    # print("False Negative: ", conf_mat[1, 0])
    #
    # # Plotting AUC ROC Curve
    # def generate_auc_roc_curve(clf, x_test):
    #     y_pred_proba = clf.predict_proba(x_test)[:, 1]
    #     fpr, tpr, thresholds = roc_curve(test_labels, y_pred_proba)
    #     auc = roc_auc_score(test_labels, y_pred_proba)
    #     plt.plot(fpr, tpr, label="AUC ROC Curve with Area Under the curve =" + str(auc))
    #     plt.legend(loc=4)
    #     plt.show()
    #
    # return generate_auc_roc_curve(model_GB, predictions)


    accuracy = accuracy_score(predictions['Potability'], predictions['Prediction'])
    precision = precision_score(predictions['Potability'], predictions['Prediction'])
    recall = recall_score(predictions['Potability'], predictions['Prediction'])
    f1 = f1_score(predictions['Potability'], predictions['Prediction'])
    confusion_matrix = pd.crosstab(
        predictions['Potability'],
        predictions['Prediction'],
        rownames=['Actual'],
        colnames=['Predicted']
    )

    # Log metrics
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=predictions['Potability'].values,
                                                       preds=predictions['Prediction'].values,
                                                       class_names=['0', '1']),
               "accuracy": accuracy,
               "precision": precision,
               "recall": recall,
               "f1_score": f1})

    return pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1]
    }), confusion_matrix