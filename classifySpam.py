import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
def aucCV(features, labels):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
    }
    model = XGBClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='roc_auc')
    grid_search.fit(features, labels)
    best_params = grid_search.best_params_
    
    model = XGBClassifier(**best_params)
    scores = cross_val_score(model, features, labels, cv=10, scoring='roc_auc')
    return scores

def predictTest(trainFeatures, trainLabels, testFeatures):
    best_params = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 4,
    }
    model = XGBClassifier(**best_params)
    
    model.fit(trainFeatures, trainLabels)
    testOutputs = model.predict_proba(testFeatures)[:, 1]
    return testOutputs

if __name__ == "__main__":
    data = np.loadtxt('spamTrain.csv', delimiter=',')
    np.random.shuffle(data)
    features = data[:, :-1]
    labels = data[:, -1]
    
    cross_val_auc = aucCV(features, labels)
    print("10-fold cross-validation mean AUC: ", np.mean(cross_val_auc))
    
    trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    testOutputs = predictTest(trainFeatures, trainLabels, testFeatures)
    
    test_auc = roc_auc_score(testLabels, testOutputs)
    print("Test set AUC: ", test_auc)
    
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    plt.subplot(1, 1, 1)
    plt.plot(np.arange(nTestExamples), testLabels[sortIndex], 'b.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Target')
    plt.show()
    plt.subplot(1, 1, 1)
    plt.plot(np.arange(nTestExamples), testOutputs[sortIndex], 'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')
    plt.show()
    
    testLabels_pred = (testOutputs > 0.5).astype(int)
    f1 = f1_score(testLabels, testLabels_pred)
    accuracy = accuracy_score(testLabels, testLabels_pred)
    print("F1 Score: ", f1)
    print("Accuracy: ", accuracy)
    conf_matrix = confusion_matrix(testLabels, testLabels_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
#please uncomment to see the heatmap of confusion matrix
# conf_matrix = confusion_matrix(testLabels, testLabels_pred)
# plt.figure(figsize=(8, 6))
# sns.set(font_scale=1.2)  
# sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 12}, fmt='g', cmap="Blues", 
#             xticklabels=["Class 0", "Class 1"],
#             yticklabels=["Class 0", "Class 1"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

