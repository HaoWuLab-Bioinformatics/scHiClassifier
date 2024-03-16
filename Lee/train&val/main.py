import pandas as pd
import xlsxwriter
from scipy import stats
import numpy as np
from method import CNN_1D_montage_val
from sklearn.model_selection import train_test_split, StratifiedKFold
import random,os, torch
from sklearn.metrics import f1_score, precision_score, balanced_accuracy_score,recall_score,matthews_corrcoef
from collections import Counter


def seed_torch(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    types = {'Astro': 449, 'Endo': 204, 'L4': 131, 'L5': 180, 'L6': 86, 'L23': 551, 'MG': 422, 'Ndnf': 144,
             'NN1': 100, 'ODC': 1243, 'OPC': 203, 'Pvalb': 134, 'Sst': 216, 'Vip': 171}
    # alpha用于计算focal_loss
    # 每个类别对应的alpha=该类别出现频率的倒数
    alpha = []
    for value in types.values():
        ds = 1 / value
        alpha.append(ds)
    # print(alpha)

    Y = []
    X = []
    resolution = 1000000
    tps = sorted(types)
    f = open('../combo_hg19.genomesize')
    index = {}
    lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        # max_len+1是指一个长度为length的染色体在resolution分辨率下能分成max_len+1块
        # 为什么要+1？因为int(10/3)=3，是向下取整的，多出来的那一截，也要做一块。
        max_len = int(int(length) / resolution)
        # index字典中index[chr_name]存的是染色体chr_name在分别率为resolution时能分出的块数
        index[chr_name] = max_len + 1
        # f.seek(0)用于将光标移到开头
    f.seek(0)
    # 关闭文件流
    f.close()
    cell_number = 0
    for type in tps:
        for c_num in range(types[type]):
            cell_number += 1
            X.append(str(cell_number))
            if type == 'Astro':
                Y.append(0)
            elif type == 'Endo':
                Y.append(1)
            elif type == 'L4':
                Y.append(2)
            elif type == 'L5':
                Y.append(3)
            elif type == 'L6':
                Y.append(4)
            elif type == 'L23':
                Y.append(5)
            elif type == 'MG':
                Y.append(6)
            elif type == 'Ndnf':
                Y.append(7)
            elif type == 'NN1':
                Y.append(8)
            elif type == 'ODC':
                Y.append(9)
            elif type == 'OPC':
                Y.append(10)
            elif type == 'Pvalb':
                Y.append(11)
            elif type == 'Sst':
                Y.append(12)
            elif type == 'Vip':
                Y.append(13)
    X = np.array(X).reshape(cell_number, 1)
    Y = np.array(Y)

    num_features = 4
    num_types = 14
    key_size, query_size, value_size = 3060, 3060, 3060
    Con_layer = [2, 2, 2, 2]
    # for testseed in [10, 541, 800, 1654, 8666]:
    random.seed(2023)
    testseed_list = random.sample(range(1, 999999), 50)

    zuhes = {1: ['nbcp3_with0', 'sbcp', 'nsicp2', 'ssicp'],
             }
    for zuhe in zuhes.values():
        # 读取特征集
        f1 = zuhe[0]
        f1_path = "../Lee_features/%s.npy" % f1
        f1_Data = np.load(f1_path, allow_pickle=True).item()  # 返回的长度为细胞数量
        f2 = zuhe[1]
        f2_path = "../Lee_features/%s.npy" % f2
        f2_Data = np.load(f2_path, allow_pickle=True).item()  # 返回的长度为细胞数量
        f3 = zuhe[2]
        f3_path = "../Lee_features/%s.npy" % f3
        f3_Data = np.load(f3_path, allow_pickle=True).item()  # 返回的长度为细胞数量
        f4 = zuhe[3]
        f4_path = "../Lee_features/%s.npy" % f4
        f4_Data = np.load(f4_path, allow_pickle=True).item()  # 返回的长度为细胞数量

        file = './train&val_result/train&val_result.xlsx'
        row = 0
        workbook = xlsxwriter.Workbook(file)
        worksheet1 = workbook.add_worksheet('111')
        worksheet1.write(0, 0, 'num_features')
        worksheet1.write(0, 1, 'num_hiddens')
        worksheet1.write(0, 2, 'ffn_num_hiddens')
        worksheet1.write(0, 3, 'num_heads')
        worksheet1.write(0, 4, 'num_layers')
        worksheet1.write(0, 5, 'dp')
        worksheet1.write(0, 6, 'use_bias')
        worksheet1.write(0, 7, 'kernel_size')
        worksheet1.write(0, 8, 'out_channels')
        worksheet1.write(0, 9, 'out_feature')
        worksheet1.write(0, 10, 'Con_layer')
        worksheet1.write(0, 11, 'linear_layer')
        worksheet1.write(0, 12, 'num_types')
        worksheet1.write(0, 13, 'focalloss_gamma')
        worksheet1.write(0, 14, 'lr')
        worksheet1.write(0, 15, 'batch_size')
        worksheet1.write(0, 16, 'val_acc')
        worksheet1.write(0, 17, 'val_micro_F1')
        worksheet1.write(0, 18, 'val_macro_F1')
        worksheet1.write(0, 19, 'val_micro_Precision')
        worksheet1.write(0, 20, 'val_macro_Precision')
        worksheet1.write(0, 21, 'val_micro_Recall')
        worksheet1.write(0, 22, 'val_macro_Recall')
        worksheet1.write(0, 23, 'val_bacc')
        for num_hiddens in [3060]:
            norm_shape = [num_features, num_hiddens]
            ffn_num_input = num_hiddens
            for ffn_num_hiddens in [int(ffn_num_input * 0.5), int(ffn_num_input * 1), int(ffn_num_input * 2)]:
                for num_heads in [20]:
                    for num_layers in [1, 2, 3, 4]:
                        for dp in [0.1, 0.3, 0.5]:
                            for use_bias in [False, True]:
                                for kernel_size in [5, 7, 10, 15]:
                                    for out_channels in [32, 64]:
                                        for out_feature in [1330, int(num_hiddens / 10), int(num_hiddens / 5),
                                                            int(num_hiddens / 2)]:
                                            for linear_layer in [1, 2, 3]:
                                                for lr in [0.1, 0.001, 0.0001, 0.00001]:
                                                    for gamma in [1, 2, 3, 4, 5]:
                                                        for batch_size in [8, 16, 32, 64]:
                                                            row += 1
                                                            sum_val_accuracy_zong = 0
                                                            sum_val_micro_F1_zong = 0
                                                            sum_val_macro_F1_zong = 0
                                                            sum_val_micro_Precision_zong = 0
                                                            sum_val_macro_Precision_zong = 0
                                                            sum_val_micro_Recall_zong = 0
                                                            sum_val_macro_Recall_zong = 0
                                                            sum_val_bacc_zong = 0
                                                            for testseed in testseed_list:
                                                                x_train, x_test, y_train, y_test = train_test_split(X,
                                                                                                                    Y,
                                                                                                                    test_size=0.2,
                                                                                                                    random_state=testseed,
                                                                                                                    stratify=Y)
                                                                model_para = [num_features, key_size, query_size,
                                                                              value_size,
                                                                              num_hiddens, norm_shape, ffn_num_input,
                                                                              ffn_num_hiddens, num_heads, num_layers,
                                                                              dp,
                                                                              use_bias,
                                                                              kernel_size, out_channels, out_feature,
                                                                              Con_layer,
                                                                              linear_layer, num_types]
                                                                print(testseed)
                                                                Folds = StratifiedKFold(n_splits=5, shuffle=True,
                                                                                        random_state=2024).split(
                                                                    x_train, y_train)
                                                                sum_val_accuracy = 0
                                                                sum_val_micro_F1 = 0
                                                                sum_val_macro_F1 = 0
                                                                sum_val_micro_Precision = 0
                                                                sum_val_macro_Precision = 0
                                                                sum_val_micro_Recall = 0
                                                                sum_val_macro_Recall = 0
                                                                sum_val_bacc = 0
                                                                for fold, (tr_idx, val_idx) in enumerate(Folds):
                                                                    tr_x, tr_y, val_x, val_y = x_train[tr_idx], y_train[
                                                                        tr_idx], x_train[val_idx], y_train[val_idx]
                                                                    val_accuracy, val_label, label, val_result_matrix = CNN_1D_montage_val(
                                                                        f1_Data, f2_Data, f3_Data, f4_Data, tr_x, tr_y,
                                                                        val_x, val_y, lr, fold, model_para, alpha,
                                                                        gamma, batch_size, testseed)
                                                                    real_label = label
                                                                    val_micro_F1 = f1_score(real_label, val_label,
                                                                                            average='micro')
                                                                    val_macro_F1 = f1_score(real_label, val_label,
                                                                                            average='macro')
                                                                    val_micro_Precision = precision_score(real_label,
                                                                                                          val_label,
                                                                                                          average='micro')
                                                                    val_macro_Precision = precision_score(real_label,
                                                                                                          val_label,
                                                                                                          average='macro')
                                                                    val_micro_Recall = recall_score(real_label,
                                                                                                    val_label,
                                                                                                    average='micro')
                                                                    val_macro_Recall = recall_score(real_label,
                                                                                                    val_label,
                                                                                                    average='macro')
                                                                    val_bacc = balanced_accuracy_score(real_label,
                                                                                                       val_label)
                                                                    sum_val_accuracy += val_accuracy
                                                                    sum_val_micro_F1 += val_micro_F1
                                                                    sum_val_macro_F1 += val_macro_F1
                                                                    sum_val_micro_Precision += val_micro_Precision
                                                                    sum_val_macro_Precision += val_macro_Precision
                                                                    sum_val_micro_Recall += val_micro_Recall
                                                                    sum_val_macro_Recall += val_macro_Recall
                                                                    sum_val_bacc += val_bacc
                                                                sum_val_accuracy_zong += sum_val_accuracy / 5
                                                                sum_val_micro_F1_zong += sum_val_micro_F1 / 5
                                                                sum_val_macro_F1_zong += sum_val_macro_F1 / 5
                                                                sum_val_micro_Precision_zong += sum_val_micro_Precision / 5
                                                                sum_val_macro_Precision_zong += sum_val_macro_Precision / 5
                                                                sum_val_micro_Recall_zong += sum_val_micro_Recall / 5
                                                                sum_val_macro_Recall_zong += sum_val_macro_Recall / 5
                                                                sum_val_bacc_zong += sum_val_bacc / 5
                                                            print("====================")
                                                            print("row:", row)
                                                            print("已完成")
                                                            print("====================")

                                                            worksheet1.write(row, 0, num_features)
                                                            worksheet1.write(row, 1, num_hiddens)
                                                            worksheet1.write(row, 2, ffn_num_hiddens)
                                                            worksheet1.write(row, 3, num_heads)
                                                            worksheet1.write(row, 4, num_layers)
                                                            worksheet1.write(row, 5, dp)
                                                            worksheet1.write(row, 6, use_bias)
                                                            worksheet1.write(row, 7, kernel_size)
                                                            worksheet1.write(row, 8, out_channels)
                                                            worksheet1.write(row, 9, out_feature)
                                                            worksheet1.write(row, 10, str(Con_layer))
                                                            worksheet1.write(row, 11, linear_layer)
                                                            worksheet1.write(row, 12, num_types)
                                                            worksheet1.write(row, 13, gamma)
                                                            worksheet1.write(row, 14, lr)
                                                            worksheet1.write(row, 15, batch_size)
                                                            worksheet1.write(row, 16, sum_val_accuracy_zong / 50)
                                                            worksheet1.write(row, 17, sum_val_micro_F1_zong / 50)
                                                            worksheet1.write(row, 18, sum_val_macro_F1_zong / 50)
                                                            worksheet1.write(row, 19, sum_val_micro_Precision_zong / 50)
                                                            worksheet1.write(row, 20, sum_val_macro_Precision_zong / 50)
                                                            worksheet1.write(row, 21, sum_val_micro_Recall_zong / 50)
                                                            worksheet1.write(row, 22, sum_val_macro_Recall_zong / 50)
                                                            worksheet1.write(row, 23, sum_val_bacc_zong / 50)
        workbook.close()


if __name__ == '__main__':
    main()