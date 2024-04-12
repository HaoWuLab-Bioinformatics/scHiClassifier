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
    """Return gpu(i) if it exists, cpu() otherwise."""
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
    f = open('../mm10.main.nochrM.chrom.sizes')
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
    return X, np.array(X).shape[0]


'''
Position-based feed-forward networks
'''
#@save
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


'''
Residual connectivity and layer normalization
'''
#@save
class AddNorm(nn.Module):
    """Layer normalization after residual joining"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

'''
Encoder.
The following EncoderBlock class contains two sublayers: multi-head self-attention and position-based feedforward networks, both of which use residual connectivity and immediately following layer normalization。
'''
#@save
class EncoderBlock(nn.Module):
    """Transformer encoder block"""
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

'''
The code for the Transformer encoder implemented below stacks num_layers instances of the EncoderBlock class.
This module, unlike the transformer model, does not encode the position of the input tensor X.
'''
class TransformerEncoder(d2l.Encoder):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        # self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, *args):
        # This differs from the transformer model in that there is no positional encoding.
        # print(self.embedding(X))
        # X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        valid_lens = None
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # blk.attention refers to MultiHeadAttention for EncoderBlock.
            # attention.attention refers to the DotProductAttention of MultiHeadAttention, so that corresponds to scaling dot product attention attention_weights
            # attention_weights has the shape (batch_size, number of queries, number of "key-value" pairs)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


def com_linearsize(linear_size, Con_layer, kernel_size):
    for i in range(Con_layer):
        # // denotes integer division, which returns the integer part of the quotient (rounded down).
        linear_size = int(((linear_size + 2 * 1 - kernel_size) / 1 + 1) // 2)
    if Con_layer == 0:
        linear_size = 0
    return linear_size

# This class is used to build the overall deep learning framework
class montage_model(nn.Module):
    def __init__(self, model_para):
        super(montage_model, self).__init__()
        num_features, key_size, query_size, value_size, num_hiddens, \
        norm_shape, ffn_num_input, ffn_num_hiddens, \
        num_heads, num_layers, dp, use_bias, kernel_size, out_channels, out_feature, Con_layer, linear_layer, num_types = model_para
        self.encoder = TransformerEncoder(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dp, use_bias)
        self.linear_layer = linear_layer
        # Because the last dimension of the TransformerEncoder output is num_hiddens
        linear_size_NBCP_init = num_hiddens
        linear_size_SBCP_init = num_hiddens
        linear_size_PSDCP_init = num_hiddens
        linear_size_SSDCP_init = num_hiddens
        self.Con_layer_NBCP, self.Con_layer_SBCP, self.Con_layer_PSDCP, self.Con_layer_SSDCP = Con_layer
        linear_size_NBCP = com_linearsize(linear_size_NBCP_init, self.Con_layer_NBCP, kernel_size)
        linear_size_SBCP = com_linearsize(linear_size_SBCP_init, self.Con_layer_SBCP, kernel_size)
        linear_size_PSDCP = com_linearsize(linear_size_PSDCP_init, self.Con_layer_PSDCP, kernel_size)
        linear_size_SSDCP = com_linearsize(linear_size_SSDCP_init, self.Con_layer_SSDCP, kernel_size)
        self.linear_size_NBCP = linear_size_NBCP
        self.linear_size_SBCP = linear_size_SBCP
        self.linear_size_PSDCP = linear_size_PSDCP
        self.linear_size_SSDCP = linear_size_SSDCP
        self.out_channels = out_channels
        if self.Con_layer_NBCP != 0:
            self.conv1_NBCP = nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
            self.bn1_NBCP = nn.BatchNorm1d(num_features=out_channels)
            self.rule1_NBCP = nn.ReLU()
            self.pool_NBCP = nn.MaxPool1d(kernel_size=2)
            self.dropout_NBCP = nn.Dropout(dp)
            self.Con_NBCP = nn.Sequential()
            for i in range(self.Con_layer_NBCP - 1):
                layer_id = str(i + 2)
                self.Con_NBCP.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_NBCP.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=out_channels))
                self.Con_NBCP.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_NBCP.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                self.Con_NBCP.add_module("drop%s" % layer_id, nn.Dropout(dp))
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
        if self.Con_layer_PSDCP != 0:
            self.conv1_PSDCP = nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
            self.bn1_PSDCP = nn.BatchNorm1d(num_features=out_channels)
            self.rule1_PSDCP = nn.ReLU()
            self.pool_PSDCP = nn.MaxPool1d(kernel_size=2)
            self.dropout_PSDCP = nn.Dropout(dp)
            self.Con_PSDCP = nn.Sequential()
            for i in range(self.Con_layer_PSDCP - 1):
                layer_id = str(i + 2)
                self.Con_PSDCP.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_PSDCP.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=out_channels))
                self.Con_PSDCP.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_PSDCP.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                self.Con_PSDCP.add_module("drop%s" % layer_id, nn.Dropout(dp))
        if self.Con_layer_SSDCP != 0:
            self.conv1_SSDCP = nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
            self.bn1_SSDCP = nn.BatchNorm1d(num_features=out_channels)
            self.rule1_SSDCP = nn.ReLU()
            self.pool_SSDCP = nn.MaxPool1d(kernel_size=2)
            self.dropout_SSDCP = nn.Dropout(dp)
            self.Con_SSDCP = nn.Sequential()
            for i in range(self.Con_layer_SSDCP - 1):
                layer_id = str(i + 2)
                self.Con_SSDCP.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_SSDCP.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=out_channels))
                self.Con_SSDCP.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_SSDCP.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                self.Con_SSDCP.add_module("drop%s" % layer_id, nn.Dropout(dp))

        if linear_layer == 1:
            self.fc = nn.Linear(in_features=out_channels * (linear_size_NBCP+linear_size_SBCP+linear_size_PSDCP+linear_size_SSDCP), out_features=num_types)
        else:
            self.fc1 = nn.Linear(in_features=out_channels * (linear_size_NBCP+linear_size_SBCP+linear_size_PSDCP+linear_size_SSDCP), out_features=out_feature)
            self.relu2 = nn.ReLU()
            self.dropout = nn.Dropout(dp)
            self.linear = nn.Sequential()
            self.out_feature = out_feature
            # When linear_layer is equal to 2, for i in range 0 is not looped.
            for i in range(linear_layer-2):
                l_layer_id = str(i + 2)
                self.linear.add_module("linear%s" % l_layer_id,nn.Linear(in_features=self.out_feature, out_features=int(self.out_feature/2)))
                self.out_feature = int(self.out_feature/2)
                self.linear.add_module("linear_relu%s" % l_layer_id,nn.ReLU())
                self.linear.add_module("linear_dropout%s" % l_layer_id, nn.Dropout(dp))
            self.fc2 = nn.Linear(in_features=self.out_feature, out_features=num_types)


    def forward(self, X):
        X = self.encoder(X)
        # X's shape is [batch_size,num_feature,2660]
        # X1's shape is [batch_size,1,2660], took the first feature
        x1 = X[:, 0:1, :]
        x2 = X[:, 1:2, :]
        x3 = X[:, 2:3, :]
        x4 = X[:, 3:4, :]
        if self.Con_layer_NBCP != 0:
            x1 = self.rule1_NBCP(self.bn1_NBCP(self.conv1_NBCP(x1)))
            x1 = self.pool_NBCP(x1)
            x1 = self.dropout_NBCP(x1)
            x1 = self.Con_NBCP(x1)
            x1 = x1.view(-1, self.out_channels * self.linear_size_NBCP)
        if self.Con_layer_SBCP != 0:
            x2 = self.rule1_SBCP(self.bn1_SBCP(self.conv1_SBCP(x2)))
            x2 = self.pool_SBCP(x2)
            x2 = self.dropout_SBCP(x2)
            x2 = self.Con_SBCP(x2)
            x2 = x2.view(-1, self.out_channels * self.linear_size_SBCP)
        if self.Con_layer_PSDCP != 0:
            x3 = self.rule1_PSDCP(self.bn1_PSDCP(self.conv1_PSDCP(x3)))
            x3 = self.pool_PSDCP(x3)
            x3 = self.dropout_PSDCP(x3)
            x3 = self.Con_PSDCP(x3)
            x3 = x3.view(-1, self.out_channels * self.linear_size_PSDCP)
        if self.Con_layer_SSDCP != 0:
            x4 = self.rule1_SSDCP(self.bn1_SSDCP(self.conv1_SSDCP(x4)))
            x4 = self.pool_SSDCP(x4)
            x4 = self.dropout_SSDCP(x4)
            x4 = self.Con_SSDCP(x4)
            x4 = x4.view(-1, self.out_channels * self.linear_size_SSDCP)

        if self.Con_layer_NBCP != 0 and self.Con_layer_SBCP != 0 and self.Con_layer_PSDCP != 0 and self.Con_layer_SSDCP != 0:
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

# This method is called by the CNN_1D_montage method when training the model
def CNN_train(epoch, model, optimizer, train_loader, loss_fn,device):
    i = 0
    for(tensors, labels) in train_loader:
        # Clear past gradients
        optimizer.zero_grad()
        labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
        labels = labels.to(device)
        tensors = tensors.to(device)

        # size = tensors.shape
        # print(size) # torch.Size([batch_size, 4, 3060])
        outputs = model(tensors)
        #  loss_fn = MultiClassFocalLossWithAlpha(alpha=alpha, gamma=gamma)
        train_loss = loss_fn(outputs, labels)
        train_loss.backward()
        optimizer.step()
        # The first argument to torch.max outputs.data
        # is to put

        # .................................... ,
        # [-1.4242, -0.8394, -1.5071, -2.2466]], grad_fn=<LogSoftmaxBackward0>)
        # become
        # tensor([[-0.8726, -1.6201, -1.5376, -1.7758], # tensor([-0.8726, -1.6201, -1.5376, -1.7758]), # tensor([-0.8726, -1.6201, -1.5376, -1.7758])
        # .................................... ,
        # # [-1.4242, -0.8394, -1.5071, -2.2466]])
        # Take the values out.
        # The second argument, 1, refers to picking the largest value by row.
        # In this case torch.max returns the element with the largest value in each row, and its index (the index of the column in which the largest element is in the row).
        # torch.max returns two tensors, the first return value _ returns the tensor consisting of the largest values in each row.
        # The second return value, prediction, is the tensor consisting of the column index of the largest element in each row.
        # We only need the second return value.
        _, prediction = torch.max(outputs.data, 1)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(tensors), len(train_loader.dataset),
                   100. * i / len(train_loader), train_loss.cpu().data))
        i = i + 1
    return model, optimizer

# This method is called by the CNN_1D_montage method when testing the model
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

# The method is used to train and test the model.
def CNN_1D_montage(f1_Data, f2_Data, f3_Data, f4_Data, x_train, y_train, X_test, y_test, lr, model_para, alpha, gamma, batch_size, epoch0):
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



    train_X = torch.cat((train_X_f1, train_X_f2, train_X_f3, train_X_f4), 1)  # Splice in the 1st dimension
    test_X = torch.cat((test_X_f1, test_X_f2, test_X_f3, test_X_f4), 1) # Splice in the 1st dimension
    # print(train_X.shape)  # torch.Size([number of cells in training set, 4, 2660])
    # print(test_X.shape)   # torch.Size([number of cells in testing set, 4, 2660])
    train_dataset = TensorDataset(train_X, torch.from_numpy(np.array(y_train).astype(int)))
    test_dataset = TensorDataset(test_X, torch.from_numpy(np.array(y_test).astype(int)))
    test_size = test_size_f1   # Number of cells in the test set
    train_loader, test_loader = load_loader(train_dataset, test_dataset, test_size, batch_size)
    model = montage_model(model_para)
    device = try_gpu(2)
    model.to(device)
    print(model)
    num_epochs = epoch0
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

