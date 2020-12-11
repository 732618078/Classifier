import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, roc_curve, auc




df = pd.read_csv("data.txt", sep="\t")
X1, y = df.values[:, 1:5], df.values[:, 0]
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X1)
X1 = scaler.fit_transform(X1)
X2 = df.values[:, 5:]
X = np.concatenate((X1, X2), axis=1)
roc_list = []
true_list = []
predict_list = []

kf = KFold(n_splits=10, shuffle=True, random_state=0)
for train, test in kf.split(X):
    train_X, test_X, train_y, test_y = X[train], X[test], y[train], y[test]
    SVM = svm.SVC(probability=True)
    param_grid = {'C':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500],
                  'kernel':('linear', 'rbf'),
                  'gamma':[0.0001, 0.0005, 0.001, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5],
                  'decision_function_shape':['ovo', 'ovr']
                  }
    grid_search = GridSearchCV(SVM, param_grid, n_jobs=10, verbose=1)
    grid_search.fit(train_X, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    SVM = grid_search.best_estimator_
    SVM.fit(train_X, train_y)
    roc_predicted = SVM.predict_proba(test_X)
    predicted = SVM.predict(test_X)
    true_list.extend(test_y.tolist())
    predict_list.extend(predicted.tolist())
    roc_list.extend(roc_predicted.tolist())


recall = recall_score(true_list, predict_list)
confusion = confusion_matrix(true_list, predict_list)
specificity = confusion[0, 0]/(confusion[0, 0] + confusion[0, 1])
accuracy = accuracy_score(true_list, predict_list)

fo = open("score.txt", "a")
fo.write("sensivity:{:.3f}, specificity:{:.3f}, accuracy:{:.3f}".format(recall, specificity, accuracy))
fo.close()


fpr1, tpr1, thresholds1 = roc_curve(true_list, [cu[1] for cu in roc_list])
roc_auc1 = auc(fpr1, tpr1)
print(roc_auc1)




pdf = PdfPages('roc.pdf')
plt.figure(figsize=(12, 12))
plt.subplots(figsize=(7, 7))
plt.plot(fpr1, tpr1, color='red', lw=2, label='combine (area=%0.3f)' %roc_auc1)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1])
plt.ylim([0.0, 1])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')
pdf.savefig()
plt.close()
pdf.close()
