import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, matthews_corrcoef,f1_score,roc_auc_score, auc,roc_curve



df = pd.read_csv("ReliefF.txt",sep="\t",index_col=0)
#print(df.head())
df = df.reset_index()


n = 2
while n < len(df.columns.tolist())+1:
    X,y=df.values[:,1:n],df.values[:,0]
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
    auc_score = auc(fpr, tpr)
    recall = recall_score(true_list, predict_list)
    confusion = confusion_matrix(true_list,predict_list)
    specificity = confusion[0,0]/(confusion[0,0]+confusion[0,1])
    accuracy = accuracy_score(true_list, predict_list)
    matthews = matthews_corrcoef(true_list, predict_list)
    f1 = f1_score(true_list,predict_list)
    roc_auc = roc_auc_score(true_list,predict_list)
    r2 = f1_score(true_list,predict_list,average='micro')    #此时r2_score用于分类问题

    fo = open("score.txt","a")
    fo.write("sensivity:{:.3f},specificity:{:.3f},accuracy:{:.3f},matthews{:.3f},f1:{:.3f},roc_auc:{:.3f},r2:{:.3f},auc_score:{:.3f}\n".format(recall,specificity,accuracy,matthews,f1,roc_auc,r2,auc_score))
    fo.close()
    print("sensivity:{:.3f},specificity:{:.3f},accuracy:{:.3f},matthews:{:.3f},f1:{:.3f},roc_auc:{:.3f},r2:{:.3f},auc_score:{:.3f}".format(recall,specificity,accuracy,matthews,f1,roc_auc,r2,auc_score))
    n += 1
