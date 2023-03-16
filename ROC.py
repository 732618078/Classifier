import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, matthews_corrcoef,f1_score,roc_auc_score, auc,roc_curve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



df = pd.read_csv("ReliefF.txt",sep="\t",index_col=0)
#print(df.head())
df = df.reset_index()


X,y=df.values[:,1:3],df.values[:,0]
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
# X = preprocessing.scale(X)

true_list = []
predict_list = []
auc_list = []
kf = KFold(n_splits=10, shuffle=True, random_state=0)
for train, test in kf.split(X):
    train_X,test_X,train_y,test_y = X[train],X[test],y[train],y[test]
    overstamp = SMOTE(random_state=0)
    SMOTE_train_X,SMOTE_train_y = overstamp.fit_sample(train_X,train_y)
    print(pd.value_counts(SMOTE_train_y,sort=True).sort_index())
    NB = GaussianNB()
    NB.fit(SMOTE_train_X,SMOTE_train_y)
    predicted = NB.predict(test_X)
    auc_predicted = NB.predict_proba(test_X)
    true_list.extend(test_y.tolist())
    predict_list.extend(predicted.tolist())
    auc_list.extend(auc_predicted.tolist())


fpr, tpr, thresholds = roc_curve(true_list, [cu[1] for cu in auc_list])
roc_auc = auc(fpr, tpr)
pdf = PdfPages('roc.pdf')
plt.figure(figsize=(12, 12))
#plt.subplots(figsize=(7,7));
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
plt.legend(loc="lower right", prop={"size":18})
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1])
plt.ylim([0.0, 1])
plt.xlabel('1 - Specificity', fontsize=20)
plt.ylabel('Sensitivity', fontsize=20)
plt.title('ROC Curve', fontsize=30)
pdf.savefig()
plt.close()
pdf.close()
