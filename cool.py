import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

def aucCV(features, labels):
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='most_frequent'))
    
    # Define the hyperparameters and their possible values
    param_grid = {
        'gradientboostingclassifier__n_estimators': [100, 200, 300],
        'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.2],
        'gradientboostingclassifier__max_depth': [3, 4, 5],
    }

    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='roc_auc')

    grid_search.fit(features, labels)

    best_params = grid_search.best_params_
    
    print("Best Hyperparameters: ", best_params)
    
    model = make_pipeline(
        SimpleImputer(missing_values=-1, strategy='most_frequent'),
        GradientBoostingClassifier(loss='log_loss', **best_params)
    )
    
    scores = cross_val_score(model, features, labels, cv=10, scoring='roc_auc')
    return scores

def predictTest(trainFeatures, trainLabels, testFeatures):
    model = make_pipeline(SimpleImputer(strategy='most_frequent'), GradientBoostingClassifier())
    best_params = {
        'n_estimators': 200,  
        'learning_rate': 0.1,  
        'max_depth': 3,  
    }
    
    model.set_params(gradientboostingclassifier__n_estimators=best_params['n_estimators'],
                    gradientboostingclassifier__learning_rate=best_params['learning_rate'],
                    gradientboostingclassifier__max_depth=best_params['max_depth'])
    
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
