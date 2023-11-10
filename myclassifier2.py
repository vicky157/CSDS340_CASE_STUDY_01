import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, confusion_matrix, auc
from sklearn.model_selection import cross_val_score, train_test_split

def aucCV(features, labels):
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='most_frequent'), GradientBoostingClassifier(loss='log_loss', learning_rate=0.01, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0))
    scores = cross_val_score(model, features, labels, cv=10, scoring='roc_auc')
    return scores

def predictTest(trainFeatures, trainLabels, testFeatures):
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='most_frequent'), GradientBoostingClassifier(loss='log_loss', learning_rate=0.01, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0))
    model.fit(trainFeatures, trainLabels)
    testOutputs = model.predict_proba(testFeatures)[:, 1]
    return testOutputs

if __name__ == "__main":
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
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(nTestExamples), testLabels[sortIndex], 'b.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Target')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(nTestExamples), testOutputs[sortIndex], 'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')

    testLabels_pred = (testOutputs > 0.5).astype(int)
    f1 = f1_score(testLabels, testLabels_pred)
    accuracy = accuracy_score(testLabels, testLabels_pred)
    print("F1 Score: ", f1)
    print("Accuracy: ", accuracy)
    conf_matrix = confusion_matrix(testLabels, testLabels_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    plt.show()
