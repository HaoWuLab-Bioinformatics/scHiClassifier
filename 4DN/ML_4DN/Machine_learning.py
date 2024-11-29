from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm,metrics
from sklearn.metrics import f1_score, precision_score, balanced_accuracy_score,recall_score,matthews_corrcoef
#用于对比的机器学习方法


def generate_bin():
    f = open('./combo_hg19.genomesize')
    index = {}
    resolution = 1000000
    lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        chr_name = chr_name
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1
        f.seek(0, 0)
    f.close()
    return index


def load_Feature_data(Feature, idX):
    index = generate_bin()
    chr_list = sorted(index.keys())
    X = []
    for cell in idX:
        feature = []
        for chr in chr_list:
            feature.append(Feature[cell[0]][chr])
        X.append(np.concatenate(feature).tolist())
    # print(np.array(X).shape)  # 414 * 3053
    # deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return X, np.array(X).shape[0]


def machinelearning_SVM(f1_Data, f2_Data, f3_Data, f4_Data, x_train, y_train, X_test, Y_test):
    train_X_f1, train_size_f1 = load_Feature_data(f1_Data, x_train)
    test_X_f1, test_size_f1 = load_Feature_data(f1_Data, X_test)
    train_X_f2, train_size_f2 = load_Feature_data(f2_Data, x_train)
    test_X_f2, test_size_f2 = load_Feature_data(f2_Data, X_test)
    train_X_f3, train_size_f3 = load_Feature_data(f3_Data, x_train)
    test_X_f3, test_size_f3 = load_Feature_data(f3_Data, X_test)
    train_X_f4, train_size_f4 = load_Feature_data(f4_Data, x_train)
    test_X_f4, test_size_f4 = load_Feature_data(f4_Data, X_test)

    Train_X = np.hstack((train_X_f1, train_X_f2, train_X_f3,train_X_f4))
    Train_Y = y_train
    Test_X = np.hstack((test_X_f1, test_X_f2, test_X_f3, test_X_f4))
    Test_Y = Y_test

    test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc = SVM(Train_X, Train_Y, Test_X, Test_Y)
    result = [test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc]
    return result

def machinelearning_LR(f1_Data, f2_Data, f3_Data, f4_Data, x_train, y_train, X_test, Y_test):
    train_X_f1, train_size_f1 = load_Feature_data(f1_Data, x_train)
    test_X_f1, test_size_f1 = load_Feature_data(f1_Data, X_test)
    train_X_f2, train_size_f2 = load_Feature_data(f2_Data, x_train)
    test_X_f2, test_size_f2 = load_Feature_data(f2_Data, X_test)
    train_X_f3, train_size_f3 = load_Feature_data(f3_Data, x_train)
    test_X_f3, test_size_f3 = load_Feature_data(f3_Data, X_test)
    train_X_f4, train_size_f4 = load_Feature_data(f4_Data, x_train)
    test_X_f4, test_size_f4 = load_Feature_data(f4_Data, X_test)

    Train_X = np.hstack((train_X_f1, train_X_f2, train_X_f3,train_X_f4))
    Train_Y = y_train
    Test_X = np.hstack((test_X_f1, test_X_f2, test_X_f3, test_X_f4))
    Test_Y = Y_test

    test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc = logistic_Reg(Train_X, Train_Y, Test_X, Test_Y)
    result = [test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc]
    return result

def machinelearning_RF(f1_Data, f2_Data, f3_Data, f4_Data, x_train, y_train, X_test, Y_test):
    train_X_f1, train_size_f1 = load_Feature_data(f1_Data, x_train)
    test_X_f1, test_size_f1 = load_Feature_data(f1_Data, X_test)
    train_X_f2, train_size_f2 = load_Feature_data(f2_Data, x_train)
    test_X_f2, test_size_f2 = load_Feature_data(f2_Data, X_test)
    train_X_f3, train_size_f3 = load_Feature_data(f3_Data, x_train)
    test_X_f3, test_size_f3 = load_Feature_data(f3_Data, X_test)
    train_X_f4, train_size_f4 = load_Feature_data(f4_Data, x_train)
    test_X_f4, test_size_f4 = load_Feature_data(f4_Data, X_test)

    Train_X = np.hstack((train_X_f1, train_X_f2, train_X_f3,train_X_f4))
    Train_Y = y_train
    Test_X = np.hstack((test_X_f1, test_X_f2, test_X_f3, test_X_f4))
    Test_Y = Y_test

    test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc = randomForest(Train_X, Train_Y, Test_X, Test_Y)
    result = [test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc]
    return result


def Search(train_model,params, Train_X,Train_Y, Test_X, Test_Y):
    # 网格寻优的训练及测试
    train_model = GridSearchCV(estimator=train_model, param_grid=params, cv=5)
    train_model.fit(Train_X, Train_Y)
    test_label_pred = train_model.predict(Test_X)
    test_acc = metrics.accuracy_score(Test_Y, test_label_pred)
    real_label = Test_Y
    test_label = test_label_pred

    micro_F1 = f1_score(real_label, test_label, average='micro')
    macro_F1 = f1_score(real_label, test_label, average='macro')
    micro_Precision = precision_score(real_label, test_label,
                                      average='micro')
    macro_Precision = precision_score(real_label, test_label,
                                      average='macro')
    micro_Recall = recall_score(real_label, test_label,
                                average='micro')
    macro_Recall = recall_score(real_label, test_label,
                                average='macro')
    bacc = balanced_accuracy_score(real_label, test_label)
    mcc = matthews_corrcoef(real_label, test_label)

    return test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc

def SVM(Train_X, Train_Y, Test_X, Test_Y):
    #SVM
    train_model = svm.SVC(probability=True)
    params = [
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
        {'kernel': ['poly'], 'C': [1, 10], 'degree': [2, 3]},
        {'kernel': ['rbf'], 'C': [1, 10, 100, 1000],
         'gamma': [1, 0.1, 0.01, 0.001]}]
    test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc = Search(train_model, params, Train_X, Train_Y, Test_X, Test_Y)

    return test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc

def logistic_Reg(Train_X, Train_Y, Test_X, Test_Y):
    #逻辑回归
    train_model = LogisticRegression()
    params = [{'C': [0.01,0.1,1,10,100]}]
    test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc = Search(train_model, params, Train_X, Train_Y, Test_X, Test_Y)

    return test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc


def randomForest(Train_X, Train_Y, Test_X, Test_Y):
    #随机森林
    train_model = RandomForestClassifier()
    params = {"n_estimators":[10,50,100,200,300,400],"max_depth":[2,4,6,10,30]}
    test_acc,  micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc = Search(train_model,params, Train_X,Train_Y, Test_X, Test_Y)

    return test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc
