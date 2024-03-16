import os, random
import math
import numpy as np
from torch import nn
from d2l import torch as d2l
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from torch.optim import Adam
from focal_loss_GPU2 import MultiClassFocalLossWithAlpha

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_bin():
    f = open("../combo_hg19.genomesize")
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
    # print(np.array(X).shape)  # 414 * 2660
    # deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return X, np.array(X).shape[0]


'''
基于位置的前馈网络
基于位置的前馈网络对序列中的所有位置的表示进行变换时使用的是同一个多层感知机（MLP），
这就是称前馈网络是基于位置的（positionwise）的原因。
在下面的实现中，输入X的形状（批量大小，时间步数或序列长度，隐单元数或特征维度）将被一个两层的感知机转换
成形状为（批量大小，时间步数，ffn_num_outputs）的输出张量。
'''
#@save
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


'''
10.7.3. 残差连接和层规范化
现在可以使用残差连接和层规范化来实现AddNorm类。暂退法也被作为正则化方法使用。
'''
#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

'''
10.7.4. 编码器
有了组成Transformer编码器的基础组件，现在可以先实现编码器中的一个层。
下面的EncoderBlock类包含两个子层：多头自注意力和基于位置的前馈网络，这两个子层都使用了残差连接和紧随的层规范化。
'''
#@save
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        X = X.to(torch.float32)
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

# 下面实现的Transformer编码器的代码中，堆叠了num_layers个EncoderBlock类的实例。由于这里使用的是值范围在-1 和 1
# 之间的固定位置编码，因此通过学习得到的输入的嵌入表示的值需要先乘以嵌入维度的平方根进行重新缩放，然后再与位置编码相加。
#@save
# 下面实现的Transformer编码器的代码中，堆叠了num_layers个EncoderBlock类的实例。由于这里使用的是值范围在-1 和 1
# 之间的固定位置编码，因此通过学习得到的输入的嵌入表示的值需要先乘以嵌入维度的平方根进行重新缩放，然后再与位置编码相加。
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        # torch.nn.Embedding(numembeddings,embeddingdim)的意思是创建一个词嵌入模型，
        # numembeddings代表一共有多少个词, embedding_dim代表你想要为每个词创建一个多少维的向量来表示它
        # 生成的嵌入的值满足均值为0，方差为1的正态分布
        # embedding = nn.Embedding(5, 3) , 那么字典中只有5个词，词向量维度为4
        # word = [[1, 2, 3],
        #         [2, 0, 4]]
        # 每个数字代表一个词，例如 {'!':0,'how':1, 'are':2, 'you':3,  'ok':4}
        # 而且这些数字的范围只能在0～4之间，因为上面定义了只有5个词
        # embedding = nn.Embedding(500000, 100)
        #     word = torch.LongTensor([[60000, 499999], [0, 9999]])
        #     c = embedding(word)
        #     print(c)
        # tensor([[[7.8601e-01,...-3.8017e-03],..]]],grad_fn=<EmbeddingBackward0>)
        # self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, *args):
        # # 因为位置编码值在-1和1之间，
        # # 而如果vocab_size和 num_hiddens过大那么生成的embedding值其实有的时候还是比较小的-2.6360e-02,1.0201e-03,
        # # 因此嵌入值需要乘以嵌入维度的平方根进行缩放，然后再与位置编码相加。
        # print(self.embedding(X))
        # X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        valid_lens = None
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # blk.attention指的是EncoderBlock的MultiHeadAttention，
            # attention.attention指的是多头注意力的DotProductAttention，所以说对应的是缩放点积注意力的attention_weights
            # attention_weights的形状为(batch_size，查询的个数，“键-值”对的个数)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


def com_linearsize(linear_size, Con_layer, kernel_size):
    for i in range(Con_layer):
        # //表示整数除法,它可以返回商的整数部分(向下取整)
        linear_size = int(((linear_size + 2 * 1 - kernel_size) / 1 + 1) // 2)
    if Con_layer == 0:
        linear_size = 0
    return linear_size


class montage_model(nn.Module):
    def __init__(self, model_para):
        super(montage_model, self).__init__()
        num_features, key_size, query_size, value_size, num_hiddens, \
        norm_shape, ffn_num_input, ffn_num_hiddens, \
        num_heads, num_layers, dp, use_bias, kernel_size, out_channels, out_feature, Con_layer, linear_layer, num_types = model_para
        self.encoder = TransformerEncoder(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dp, use_bias)
        self.linear_layer = linear_layer
        # 因为Transformer输出的最后一维的维度就是num_hiddens
        linear_size_NBCP3_init = num_hiddens
        linear_size_SBCP_init = num_hiddens
        linear_size_NSICP2_init = num_hiddens
        linear_size_SSICP_init = num_hiddens
        self.Con_layer_NBCP3, self.Con_layer_SBCP, self.Con_layer_NSICP2, self.Con_layer_SSICP = Con_layer
        linear_size_NBCP3 = com_linearsize(linear_size_NBCP3_init, self.Con_layer_NBCP3, kernel_size)
        linear_size_SBCP = com_linearsize(linear_size_SBCP_init, self.Con_layer_SBCP, kernel_size)
        linear_size_NSICP2 = com_linearsize(linear_size_NSICP2_init, self.Con_layer_NSICP2, kernel_size)
        linear_size_SSICP = com_linearsize(linear_size_SSICP_init, self.Con_layer_SSICP, kernel_size)
        self.linear_size_NBCP3 = linear_size_NBCP3
        self.linear_size_SBCP = linear_size_SBCP
        self.linear_size_NSICP2 = linear_size_NSICP2
        self.linear_size_SSICP = linear_size_SSICP
        self.out_channels = out_channels
        if self.Con_layer_NBCP3 != 0:
            self.conv1_NBCP3 = nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
            self.bn1_NBCP3 = nn.BatchNorm1d(num_features=out_channels)
            self.rule1_NBCP3 = nn.ReLU()
            self.pool_NBCP3 = nn.MaxPool1d(kernel_size=2)
            self.dropout_NBCP3 = nn.Dropout(dp)
            self.Con_NBCP3 = nn.Sequential()
            for i in range(self.Con_layer_NBCP3 - 1):
                layer_id = str(i + 2)
                self.Con_NBCP3.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_NBCP3.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=out_channels))
                self.Con_NBCP3.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_NBCP3.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                self.Con_NBCP3.add_module("drop%s" % layer_id, nn.Dropout(dp))
        if self.Con_layer_SBCP != 0:
            self.conv1_SBCP = nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
            self.bn1_SBCP = nn.BatchNorm1d(num_features=out_channels)
            self.rule1_SBCP = nn.ReLU()
            self.pool_SBCP = nn.MaxPool1d(kernel_size=2)
            self.dropout_SBCP = nn.Dropout(dp)
            self.Con_SBCP = nn.Sequential()
            for i in range(self.Con_layer_SBCP - 1):
                layer_id = str(i + 2)
                self.Con_SBCP.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_SBCP.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=out_channels))
                self.Con_SBCP.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_SBCP.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                self.Con_SBCP.add_module("drop%s" % layer_id, nn.Dropout(dp))
        if self.Con_layer_NSICP2 != 0:
            self.conv1_NSICP2 = nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
            self.bn1_NSICP2 = nn.BatchNorm1d(num_features=out_channels)
            self.rule1_NSICP2 = nn.ReLU()
            self.pool_NSICP2 = nn.MaxPool1d(kernel_size=2)
            self.dropout_NSICP2 = nn.Dropout(dp)
            self.Con_NSICP2 = nn.Sequential()
            for i in range(self.Con_layer_NSICP2 - 1):
                layer_id = str(i + 2)
                self.Con_NSICP2.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_NSICP2.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=out_channels))
                self.Con_NSICP2.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_NSICP2.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                self.Con_NSICP2.add_module("drop%s" % layer_id, nn.Dropout(dp))
        if self.Con_layer_SSICP != 0:
            self.conv1_SSICP = nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
            self.bn1_SSICP = nn.BatchNorm1d(num_features=out_channels)
            self.rule1_SSICP = nn.ReLU()
            self.pool_SSICP = nn.MaxPool1d(kernel_size=2)
            self.dropout_SSICP = nn.Dropout(dp)
            self.Con_SSICP = nn.Sequential()
            for i in range(self.Con_layer_SSICP - 1):
                layer_id = str(i + 2)
                self.Con_SSICP.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_SSICP.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=out_channels))
                self.Con_SSICP.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_SSICP.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                self.Con_SSICP.add_module("drop%s" % layer_id, nn.Dropout(dp))

        if linear_layer == 1:
            self.fc = nn.Linear(in_features=out_channels * (linear_size_NBCP3+linear_size_SBCP+linear_size_NSICP2+linear_size_SSICP), out_features=num_types)
        else:
            self.fc1 = nn.Linear(in_features=out_channels * (linear_size_NBCP3+linear_size_SBCP+linear_size_NSICP2+linear_size_SSICP), out_features=out_feature)
            self.relu2 = nn.ReLU()
            self.dropout = nn.Dropout(dp)
            self.linear = nn.Sequential()
            self.out_feature = out_feature
            # 当linear_layer等于2时,for i in range 0 是不循环的
            for i in range(linear_layer-2):
                l_layer_id = str(i + 2)
                self.linear.add_module("linear%s" % l_layer_id,nn.Linear(in_features=self.out_feature, out_features=int(self.out_feature/2)))
                self.out_feature = int(self.out_feature/2)
                self.linear.add_module("linear_relu%s" % l_layer_id,nn.ReLU())
                self.linear.add_module("linear_dropout%s" % l_layer_id, nn.Dropout(dp))
            self.fc2 = nn.Linear(in_features=self.out_feature, out_features=num_types)


    def forward(self, X):
        X = self.encoder(X)
        # X的shape为[batch_size,num_feature,2660]
        # X1的shape为[batch_size,1,2660],取了第一个特征
        x1 = X[:, 0:1, :]
        x2 = X[:, 1:2, :]
        x3 = X[:, 2:3, :]
        x4 = X[:, 3:4, :]
        if self.Con_layer_NBCP3 != 0:
            x1 = self.rule1_NBCP3(self.bn1_NBCP3(self.conv1_NBCP3(x1)))
            x1 = self.pool_NBCP3(x1)
            x1 = self.dropout_NBCP3(x1)
            x1 = self.Con_NBCP3(x1)
            x1 = x1.view(-1, self.out_channels * self.linear_size_NBCP3)
        if self.Con_layer_SBCP != 0:
            x2 = self.rule1_SBCP(self.bn1_SBCP(self.conv1_SBCP(x2)))
            x2 = self.pool_SBCP(x2)
            x2 = self.dropout_SBCP(x2)
            x2 = self.Con_SBCP(x2)
            x2 = x2.view(-1, self.out_channels * self.linear_size_SBCP)
        if self.Con_layer_NSICP2 != 0:
            x3 = self.rule1_NSICP2(self.bn1_NSICP2(self.conv1_NSICP2(x3)))
            x3 = self.pool_NSICP2(x3)
            x3 = self.dropout_NSICP2(x3)
            x3 = self.Con_NSICP2(x3)
            x3 = x3.view(-1, self.out_channels * self.linear_size_NSICP2)
        if self.Con_layer_SSICP != 0:
            x4 = self.rule1_SSICP(self.bn1_SSICP(self.conv1_SSICP(x4)))
            x4 = self.pool_SSICP(x4)
            x4 = self.dropout_SSICP(x4)
            x4 = self.Con_SSICP(x4)
            x4 = x4.view(-1, self.out_channels * self.linear_size_SSICP)

        if self.Con_layer_NBCP3 != 0 and self.Con_layer_SBCP != 0 and self.Con_layer_NSICP2 != 0 and self.Con_layer_SSICP != 0:
            x = torch.cat((x1, x2, x3, x4), 1)

        if self.linear_layer == 1:
            x = self.fc(x)
        else:# Decay:492  Inscore:489
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu2(x)
            x = self.linear(x)
            x = self.fc2(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x


def load_loader(train_dataset, test_dataset, test_size, batch_size):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(dataset=test_dataset,
                            batch_size=test_size,
                            shuffle=False)
    return train_loader, val_loader

def CNN_train(epoch, model, optimizer, train_loader, loss_fn,device):
    i = 0
    for(tensors, labels) in train_loader:
        # 清空过往梯度
        optimizer.zero_grad()
        labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
        labels = labels.to(device)
        tensors = tensors.to(device)

        # size = tensors.shape
        # print(size) # torch.Size([batch_size, 4, 2660])
        outputs = model(tensors)
        #  loss_fn = MultiClassFocalLossWithAlpha(alpha=alpha, gamma=gamma)
        train_loss = loss_fn(outputs, labels)
        train_loss.backward()
        optimizer.step()
        # torch.max的第一个参数outputs.data
        # 是把
        # tensor([[-0.8726, -1.6201, -1.5376, -1.7758],
        #         ....................................,
        # [-1.4242, -0.8394, -1.5071, -2.2466]], grad_fn=<LogSoftmaxBackward0>)
        # 变成
        # tensor([[-0.8726, -1.6201, -1.5376, -1.7758],
        #         ....................................,
        #         # [-1.4242, -0.8394, -1.5071, -2.2466]])
        # 把值取出来。
        # 第二个参数1指的是按照行进行选最大值，
        # 此时torch.max返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        # torch.max共返回两个tensor，第一个返回值_返回的是每一行中的最大值所组成的tensor。
        # 第二个返回值prediction是每一行中的最大元素在这一行的列索引所组成的tensor。
        # 我们只需要第二个返回值就够了
        _, prediction = torch.max(outputs.data, 1)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(tensors), len(train_loader.dataset),
                   100. * i / len(train_loader), train_loss.cpu().data))
        # 之前没有下面这一行代码，而且i=0也放在了for循环的内部
        i = i + 1
    return model, optimizer


def CNN_test(epoch,model, test_loader, loss_fn, test_size, device):
    i = 0
    for (tensors, labels) in test_loader:
        labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
        labels = labels.to(device)
        tensors = tensors.to(device)
        outputs = model(tensors)
        test_loss = loss_fn(outputs, labels)
        test_result_matrix0 = outputs
        test_result_matrix = test_result_matrix0.cpu().detach().numpy()
        _, prediction = torch.max(outputs.data, 1)
        label_pred = prediction.cpu().numpy()
        label = labels.data.cpu().numpy()

        prediction_num = int(torch.sum(prediction == labels.data))
        test_accuracy = prediction_num / test_size
        print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(tensors), len(test_loader.dataset),
                   100. * i / len(test_loader), test_loss.cpu().data ))
        i = i+1
    return label_pred, label, test_loss, model, test_accuracy, test_result_matrix

#
def CNN_1D_montage(f1_Data, f2_Data, f3_Data, f4_Data, x_train, y_train, X_test, y_test, lr, model_para, alpha, gamma, batch_size):
    train_X_f1, train_size_f1 = load_Feature_data(f1_Data, x_train)
    test_X_f1, test_size_f1 = load_Feature_data(f1_Data, X_test)
    train_X_f2, train_size_f2 = load_Feature_data(f2_Data, x_train)
    test_X_f2, test_size_f2 = load_Feature_data(f2_Data, X_test)
    train_X_f3, train_size_f3 = load_Feature_data(f3_Data, x_train)
    test_X_f3, test_size_f3 = load_Feature_data(f3_Data, X_test)
    train_X_f4, train_size_f4 = load_Feature_data(f4_Data, x_train)
    test_X_f4, test_size_f4 = load_Feature_data(f4_Data, X_test)

    train_X_f1 = torch.unsqueeze(torch.from_numpy(np.array(train_X_f1).astype(float)), dim=1)
    test_X_f1 = torch.unsqueeze(torch.from_numpy(np.array(test_X_f1).astype(float)), dim=1)
    train_X_f2 = torch.unsqueeze(torch.from_numpy(np.array(train_X_f2).astype(float)), dim=1)
    test_X_f2 = torch.unsqueeze(torch.from_numpy(np.array(test_X_f2).astype(float)), dim=1)
    train_X_f3 = torch.unsqueeze(torch.from_numpy(np.array(train_X_f3).astype(float)), dim=1)
    test_X_f3 = torch.unsqueeze(torch.from_numpy(np.array(test_X_f3).astype(float)), dim=1)
    train_X_f4 = torch.unsqueeze(torch.from_numpy(np.array(train_X_f4).astype(float)), dim=1)
    test_X_f4 = torch.unsqueeze(torch.from_numpy(np.array(test_X_f4).astype(float)), dim=1)
    print(train_X_f1.shape)
    # print(train_X_f1.shape)  # torch.Size([训练集细胞个数, 1, 3053])
    # print(test_X_f1.shape)    # torch.Size([测试集细胞个数, 1, 3053])
    #  train_X_f1的Size([500, 1, 3053])，由于3053在进多头自注意力的时候不能被20整除，所以需要补7个0到3060
    zeros_train_f1 = torch.zeros(train_size_f1, 1, 7)
    zeros_test_f1 = torch.zeros(test_size_f1, 1, 7)
    zeros_train_f2 = torch.zeros(train_size_f2, 1, 7)
    zeros_test_f2 = torch.zeros(test_size_f2, 1, 7)
    zeros_train_f3 = torch.zeros(train_size_f3, 1, 7)
    zeros_test_f3 = torch.zeros(test_size_f3, 1, 7)
    zeros_train_f4 = torch.zeros(train_size_f4, 1, 7)
    zeros_test_f4 = torch.zeros(test_size_f4, 1, 7)
    train_X_f1 = torch.cat((train_X_f1, zeros_train_f1), 2)
    test_X_f1 = torch.cat((test_X_f1, zeros_test_f1), 2)
    train_X_f2 = torch.cat((train_X_f2, zeros_train_f2), 2)
    test_X_f2 = torch.cat((test_X_f2, zeros_test_f2), 2)
    train_X_f3 = torch.cat((train_X_f3, zeros_train_f3), 2)
    test_X_f3 = torch.cat((test_X_f3, zeros_test_f3), 2)
    train_X_f4 = torch.cat((train_X_f4, zeros_train_f4), 2)
    test_X_f4 = torch.cat((test_X_f4, zeros_test_f4), 2)

    train_X = torch.cat((train_X_f1, train_X_f2, train_X_f3, train_X_f4), 1)  # 在1维进行拼接
    test_X = torch.cat((test_X_f1, test_X_f2, test_X_f3, test_X_f4), 1)  # 在1维进行拼接
    # print(train_X.shape)  # torch.Size([415, 4, 2660])
    # print(test_X.shape)   # torch.Size([103, 4, 2660])
    train_dataset = TensorDataset(train_X, torch.from_numpy(np.array(y_train).astype(int)))
    test_dataset = TensorDataset(test_X, torch.from_numpy(np.array(y_test).astype(int)))
    test_size = test_size_f1  # 103
    train_loader, test_loader = load_loader(train_dataset, test_dataset, test_size, batch_size)
    model = montage_model(model_para)
    device = try_gpu(2)
    model.to(device)
    # print(model)
    num_epochs = 420
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MultiClassFocalLossWithAlpha(alpha=alpha, gamma=gamma)
    for epoch in range(num_epochs):
        model.train()
        model, optimizer = CNN_train(epoch, model, optimizer, train_loader, loss_fn,device)
        model.eval()
        test_label, label, test_loss, model, test_accuracy, test_result_matrix = CNN_test(epoch, model, test_loader,
                                                                          loss_fn, test_size_f1,device)
    torch.cuda.empty_cache()

    return test_accuracy, test_label, label, test_result_matrix

