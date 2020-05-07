#Model include seq
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import shutil

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir + 'checkpoint_seq.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + 'best_model_seq.pt'
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'], checkpoint['best_accuracy']

class Batcher:
    def __init__(self, num_items, batch_size, seed=0):
        self.indices = np.arange(num_items)
        self.num_items = num_items
        self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)
        self.rnd.shuffle(self.indices)
        self.ptr = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.ptr + self.batch_size > self.num_items:
            self.rnd.shuffle(self.indices)
            self.ptr = 0
            raise StopIteration  # exit calling for-loop
        else:
            result = self.indices[self.ptr:self.ptr+self.batch_size]
            self.ptr += self.batch_size
            return result

class Network_epi(nn.Module):
    def __init__(self):
        super(Network_epi, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4)
        self.pool3 = nn.MaxPool1d(8, stride=4)
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4)
        self.pool4 = nn.MaxPool1d(8, stride=4)
        self.bn1 = nn.BatchNorm1d(num_features=6*(16665+12498))
        self.drop1 = nn.Dropout(0.5)
        self.lstm = nn.LSTM(6*(16665+12498), 256*6, num_layers=1, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(num_features=2*6*256)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2*6*256, 2*256)
        self.fc2 = nn.Linear(2*256, 128)
        self.fc3 = nn.Linear(128, 2)
    def forward(self, input_):
        x = input_[0]
        y = input_[1]
        x1 = x[:, :1,:,:]
        x2 = x[:, 1:,:,:]
        y1 = y[:, :1,:,:]
        y2 = y[:, 1:,:,:]
        x1 = self.pool1(F.relu(self.conv1(x1)))
        x2 = self.pool2(F.relu(self.conv2(x2)))
        y1 = F.relu(self.conv3(y1))
        y1 = y1[:,:,-1,:]
        y1 = self.pool3(y1)
        y2 = F.relu(self.conv4(y2))
        y2 = y2[:,:,-1,:]
        y2 = self.pool3(y2)
        x = torch.cat((x1,x2),1)
        x = x.view(x.size(0), -1)
        y = torch.cat((y1,y2),1)
        y = y.view(y.size(0), -1)
        x = torch.cat((x,y),1)
        x = x.view(x.size(0), -1)
        x = self.drop1(self.bn1(x))
        x = x[:, None]
        x, states = self.lstm(x)
        x = x.view(x.size(0), -1)
        x = self.drop2(self.bn2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def predict(self,x):
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

model = Network_epi()

#Region 1 for train
ep1 = np.load("GM12878_50kb_sig0.05_chr21_train_region1_epi_standardized.npy")
seq1 = np.load("GM12878_chr21_50kb_known_pairs_train_region1_Seq.npz")
seq1 = seq1['sequence']
#Region 1 for test
ep1_test = np.load("GM12878_50kb_sig0.05_chr21_test_region1_epi_standardized.npy")
ep1_test = torch.from_numpy(ep1_test).type(torch.FloatTensor)
ep1_test = ep1_test[:, None]
seq1_test = np.load("GM12878_chr21_50kb_known_pairs_test_region1_Seq.npz")
seq1_test = seq1_test['sequence']
seq1_test = seq1_test.reshape(len(seq1_test),50000,4)
seq1_test = seq1_test.swapaxes(2,1)
seq1_test = torch.from_numpy(seq1_test).type(torch.FloatTensor)
seq1_test = seq1_test[:, None]
#Region 2 for train
ep2 = np.load("GM12878_50kb_sig0.05_chr21_train_region2_epi_standardized.npy")
seq2 = np.load("GM12878_chr21_50kb_known_pairs_train_region2_Seq.npz")
seq2 = seq2['sequence']
#Region 2 for test
ep2_test = np.load("GM12878_50kb_sig0.05_chr21_test_region2_epi_standardized.npy")
ep2_test = torch.from_numpy(ep2_test).type(torch.FloatTensor)
ep2_test = ep2_test[:, None]
seq2_test = np.load("GM12878_chr21_50kb_known_pairs_test_region1_Seq.npz")
seq2_test = seq2_test['sequence']
seq2_test = seq2_test.reshape(len(seq2_test),50000,4)
seq2_test = seq2_test.swapaxes(2,1)
seq2_test = torch.from_numpy(seq2_test).type(torch.FloatTensor)
seq2_test = seq2_test[:, None]

ep_test = torch.cat((ep1_test,ep2_test),1)
seq_test = torch.cat((seq1_test,seq2_test),1)
input_test = [ep_test,seq_test]

#Label for train
y = pd.read_csv("/ysm-gpfs/pi/zhao/xz473/DeepTact/data/split/GM12878_50kb_sig0.05_chr21_train.csv")
y = y['label']
y = y.values

#Label for test
y_test = pd.read_csv("/ysm-gpfs/pi/zhao/xz473/DeepTact/data/GM12878_50kb_sig0.05_chr21_test.csv")
y_test = y_test['label']
y_test = y_test.values
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

lrn_rate = 0.00001
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lrn_rate)
max_epochs = 100
batch_size = 256
n_items = len(ep1)
best_accuracy = 0.5
checkpoint_dir = "/ysm-gpfs/pi/zhao/xz473/DeepTact/data/"
best_model_dir = "/ysm-gpfs/pi/zhao/xz473/DeepTact/data/"

ckp_path = checkpoint_dir+'checkpoint_seq.pt'
model, optimizer, start_epoch, best_accuracy = load_ckp(ckp_path, model, optimizer)
for t in range(start_epoch, max_epochs):
    batcher = Batcher(n_items, batch_size)
    if t % 5 == 0:
        acc_val = accuracy_score(model.predict(input_test),y_test)
        # Get bool not ByteTensor
        is_best = bool(acc_val > best_accuracy)
        # Get greater Tensor to keep track best acc
        best_accuracy = max(acc_val, best_accuracy)
        checkpoint = {
            'epoch': t + 1,
            'best_accuracy': best_accuracy,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}
        save_ckp(checkpoint, is_best, checkpoint_dir, best_model_dir)
        print('%s iter %d, lr = %.2e, validation acc: %.2f' % ("Adam", t, lrn_rate, acc_val))  
    for occr in batcher:
        ep1_train = ep1[occr,:,:]
        ep1_train = torch.from_numpy(ep1_train).type(torch.FloatTensor)
        ep1_train = ep1_train[:, None]
        ep2_train = ep2[occr,:,:]
        ep2_train = torch.from_numpy(ep2_train).type(torch.FloatTensor)
        ep2_train = ep2_train[:, None]
        ep_train = torch.cat((ep1_train,ep2_train),1)
        seq1_train = seq1[occr,:]
        seq1_train = seq1_train.reshape(len(seq1_train),50000,4)
        seq1_train = seq1_train.swapaxes(2,1)
        seq1_train = torch.from_numpy(seq1_train).type(torch.FloatTensor)
        seq1_train = seq1_train[:, None]
        seq2_train = seq2[occr,:]
        seq2_train = seq2_train.reshape(len(seq2_train),50000,4)
        seq2_train = seq2_train.swapaxes(2,1)
        seq2_train = torch.from_numpy(seq2_train).type(torch.FloatTensor)
        seq2_train = seq2_train[:, None]
        seq_train = torch.cat((seq1_train,seq2_train),1)
        input_train = [ep_train,seq_train]
        y_train = y[occr]
        y_train = torch.from_numpy(y_train).type(torch.LongTensor)
        optimizer.zero_grad()
        y_pred = model(input_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
