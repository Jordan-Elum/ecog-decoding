"""
@authors: Riya Jain, Jordan Elum

This script reads in ECoG data for a single subject, shapes the data for this CNN framework, splits the data
into train and test sets, normalizes the data, passes the data into a CNN model, trains the CNN model, evaluates the model,
and plots the prediction accuracy, loss, and confusion matrix.
May need to run 'pip install torch' and 'pip install torch-summary' and 'pip install torchmetrics'

adapted from: https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/tutorial/4-2%20Classification%20based%20on%20spatial%20feature%20(CNN).ipynb
"""
#%% import libraries and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torchsummary import summary
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report

#%% get face house data set for a single subject
subject = '3'
fname = input('Enter the absolute path of the subject_3_passive_task_braodb_full_df.npy file (no quotes): ')
data_subject = np.load(fname, allow_pickle=True)
save_path = input('Enter the absolute path of the directory in which the output data should be saved (no quotes): ')

#%%  load dataset
dataset_1 = data_subject
print('dataset_1 shape:', dataset_1.shape)
# check if a GPU is available
with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
    print('We are using %s now.' %device)
else:
    device = torch.device("cpu")

# remove instance with label==10 (rest)
removed_label = [2,3,4,5,6,7,8,9,10]  #2,3,4,5,
for ll in removed_label:
     id = dataset_1[:, -1]!=ll
     dataset_1 = dataset_1[id]
print('dataset_1 shape: ', dataset_1.shape)

def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    y_ = [int(xx) for xx in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

#%% data segmentation
# n_class = int(11-len(removed_label))  # 0~9 classes ('10:rest' is not considered)
n_class = int(2)

# no_feature = 64  # the number of the features
no_feature =  dataset_1.shape[1]-1 # the number of the features

segment_length = 16  # selected time window; 16=160*0.1

LR = 0.005  # learning rate
EPOCH = 101
n_hidden = 128  # number of neurons in hidden layer
l2 = 0.01  # the coefficient of l2-norm regularization

def extract(input, n_classes, n_fea, time_window, moving):
    xx = input[:, :n_fea]
    yy = input[:, n_fea:n_fea + 1]
    new_x = []
    new_y = []
    number = int((xx.shape[0] / moving) - 1)
    for i in range(number):
        ave_y = np.average(yy[int(i * moving):int(i * moving + time_window)])
        if ave_y in range(n_classes + 1):
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(ave_y)
        else:
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(0)

    new_x = np.array(new_x)
    new_x = new_x.reshape([-1, n_fea * time_window])
    new_y = np.array(new_y)
    new_y.shape = [new_y.shape[0], 1]
    data = np.hstack((new_x, new_y))
    # data = np.vstack((data, data[-1]))  # add the last sample again, to make the sample number round
    return data

data_seg = extract(dataset_1, n_classes=n_class, n_fea=no_feature, time_window=segment_length, moving=(segment_length/2))  # 50% overlapping
print('After segmentation, the shape of the data:', data_seg.shape)

# split training and test data
no_longfeature = no_feature*segment_length
data_seg_feature = data_seg[:, :no_longfeature]
data_seg_label = data_seg[:, no_longfeature:no_longfeature+1]
train_feature, test_feature, train_label, test_label = train_test_split(data_seg_feature, data_seg_label,test_size=0.2, shuffle=True)

# normalization
# before normalize reshape data back to raw data shape
train_feature_2d = train_feature.reshape([-1, no_feature])
test_feature_2d = test_feature.reshape([-1, no_feature])

scaler1 = StandardScaler().fit(train_feature_2d)
train_fea_norm1 = scaler1.transform(train_feature_2d) # normalize the training data
test_fea_norm1 = scaler1.transform(test_feature_2d) # normalize the test data
print('After normalization, the shape of training feature:', train_fea_norm1.shape,
      '\nAfter normalization, the shape of test feature:', test_fea_norm1.shape)

# after normalization, reshape data to 3d in order to feed in to LSTM
train_fea_norm1 = train_fea_norm1.reshape([-1, segment_length, no_feature])
test_fea_norm1 = test_fea_norm1.reshape([-1, segment_length, no_feature])
print('After reshape, the shape of training feature:', train_fea_norm1.shape,
      '\nAfter reshape, the shape of test feature:', test_fea_norm1.shape)
BATCH_size = test_fea_norm1.shape[0] # use test_data as batch size

#%% feed data into dataloader
train_fea_norm1 = torch.tensor(train_fea_norm1)
train_fea_norm1 = torch.unsqueeze(train_fea_norm1, dim=1).type('torch.FloatTensor').to(device)
# print(train_fea_norm1.shape)
train_label = torch.tensor(train_label.flatten()).to(device)
train_data = Data.TensorDataset(train_fea_norm1, train_label)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_size, shuffle=False)

test_fea_norm1 = torch.tensor(test_fea_norm1)
test_fea_norm1 = torch.unsqueeze(test_fea_norm1, dim=1).type('torch.FloatTensor').to(device)
test_label = torch.tensor(test_label.flatten()).to(device)

#%% initialize CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(2,4),
                stride=1,
                padding= (1,2)  #([1,2]-1)/2,
            ),
            nn.ReLU(),
            nn.MaxPool2d((2,4))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (2,2), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.fc = nn.Linear(4*8*32, 128)  # 64*2*4
        self.out = nn.Linear(128, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = F.dropout(x, 0.2)

        output = self.out(x)
        return output, x
    
cnn = CNN()
cnn.to(device)
print(cnn)
print('model for subject: ', subject)
summary(cnn, (1, 16, no_feature))

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=l2)
loss_func = nn.CrossEntropyLoss()

best_acc = []
best_auc = []
train_accu = []

prediction_y = []
true_y = []
loss_fn = []
test_set_acc = []

#%% training and testing
i = 0
start_time = time.perf_counter()
for epoch in range(EPOCH):
    for step, (train_x, train_y) in enumerate(train_loader):

        output = cnn(train_x)[0]  # CNN output of training data
        loss = loss_func(output, train_y.long())  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        loss_fn += [loss.data.cpu().numpy()]  # calculate loss function on training set w respect to gradient descent (learn rate)
    if epoch % 10 == 0:
        test_output = cnn(test_fea_norm1)[0]  # CNN output of test data
        test_loss = loss_func(test_output, test_label.long())

        test_y_score = one_hot(test_label.data.cpu().numpy())  # .cpu() can be removed if your device is cpu.
        pred_score = F.softmax(test_output, dim=1).data.cpu().numpy()  # normalize the output
        auc_score = roc_auc_score(test_y_score, pred_score)

        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        pred_train = torch.max(output, 1)[1].data.cpu().numpy()

        test_acc = accuracy_score(test_label.data.cpu().numpy(), pred_y)
        train_acc = accuracy_score(train_y.data.cpu().numpy(), pred_train)
        train_accu.append(train_acc)
        print('CNN output of training data:  ', output, '\n',  '| cross entropy loss:   ', loss, '\n',  '| CNN output of test data test output :  ',
              test_output,'\n',  '| test_loss ', test_loss, '\n', '| test y score ', test_y_score,'\n',  '| pred score (normalize the output softmax) ', pred_score,
             '\n',  '| pred_y ', pred_y, ' | pred train:  ',pred_train)
        print('Epoch: ', epoch,  '|train loss: %.4f' % loss.item(),
              ' train ACC: %.4f' % train_acc, '| test loss: %.4f' % test_loss.item(), 'test ACC: %.4f' % test_acc, '| AUC: %.4f' % auc_score)
        best_acc.append(test_acc)
        # best_auc.append(auc_score)
        test_set_acc += [test_acc]
        
#%% plot Prediction Accuracy, Cross Entropy Loss, Confusion Matrix, Prediction vs True Labels over time
plt.figure(figsize=(7,8), dpi = 1200)
plt.plot(range(len(test_set_acc)), test_set_acc)
plt.ylabel('Accuracy')
plt.xlabel('Training epoch')
plt.title('Prediction Accuracy across Training for Subject '+str(subject))
plt.savefig(save_path + '\\' +  'Prediction Accuracy' + '.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(7,8), dpi = 1200)
plt.plot(loss_fn)
plt.title('Loss Function for Subject '+str(subject))
plt.ylabel('Cross entropy loss')
plt.xlabel('Iterations of Gradient Descent')
plt.savefig(save_path + '\\' +  'Loss Function for Subject' + '.png', bbox_inches='tight')
plt.show()

metric = MulticlassConfusionMatrix(num_classes=2)
#confusion matrix how many of array elemts were correctly classified
array_cm = metric(torch.tensor(pred_y), test_label.data.cpu()) #
array_cm # confusion matrix
#plot percentages 
plt.figure(figsize=(8,6), dpi = 1200)
array_hm = array_cm.data.cpu().numpy()/np.sum(array_cm.data.cpu().numpy(),axis=1)
#heatmap to represent confusion matrix
sns.heatmap(array_hm,cmap='RdYlGn',linewidths=0.30, annot = True,vmin=0,vmax=1)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for '+ str(subject))
plt.savefig(save_path + '\\' +  'Confusion Matrix' + '.png', bbox_inches='tight')
plt.show()

#plot predicted vs true labels over time samples in dataset 
current_time = time.perf_counter()
running_time = current_time - start_time
print(classification_report(test_label.data.cpu().numpy(), pred_y))
# print('BEST TEST ACC: {}, AUC: {}'.format(max(best_acc), max(best_auc)))
print("Total Running Time: {} seconds".format(round(running_time, 2)))
