# Code in cnn_fn_pytorch.py
from __future__ import print_function, division
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import argparse
from time import time
from load_data import *
from torch.optim import lr_scheduler
from visdom import Visdom



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
                help="path to the input file")
ap.add_argument("-n", "--epochs", default=25,
                help="epochs for train")
ap.add_argument("-b", "--batchsize", default=4,
                help="batch size for train")
ap.add_argument("-r", "--resume", default='111',
                help="file for re-train")
ap.add_argument("-w", "--writeFile", default='newmodel.out',
                help="file for output")
args = vars(ap.parse_args())

use_gpu = torch.cuda.is_available()
print (use_gpu)

numClasses = 8
imgSize = (480, 480)
batchSize = int(args["batchsize"]) if use_gpu else 8
modelFolder = 'newmodel/'
storeName = modelFolder + 'newmodel.pth'
if not os.path.isdir(modelFolder):
    os.mkdir(modelFolder)

epochs = int(args["epochs"])
#   initialize the output file
with open(args['writeFile'], 'wb') as outF:
    pass


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class wR2(nn.Module):
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )

        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )

        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )

        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,

            hidden5,

            hidden7,
            hidden8,
            hidden9,

        )
        self.classifier = nn.Sequential(
            nn.Linear(9600, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x
def eval(model, test_dirs):
    count, error, correct = 0, 0, 0
    dst = labelTestDataLoader(test_dirs, imgSize)
    testloader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=0)
    start = time()
    for i, (XI, labels, ims) in enumerate(testloader):
        count += 1
        YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
        if use_gpu:
            x = Variable(XI.cuda(0))
        else:
            x = Variable(XI)
        # Forward pass: Compute predicted y by passing x to the model

        fps_pred, y_pred = model(x)

        outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
        labelPred = [t[0].index(max(t[0])) for t in outputY]

        #   compare YI, outputY
        try:
            if isEqual(labelPred, YI[0]) == 7:
                correct += 1
            else:
                #print (correct)
                pass
        except:
            error += 1
    return count, correct, error, float(correct) / count, (time() - start) / count


epoch_start = 0
resume_file = str(args["resume"])
if not resume_file == '111':
    # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
    if not os.path.isfile(resume_file):
        print ("fail to load existed model! Existing ...")
        exit(0)
    print ("Load existed model! %s" % resume_file)
    model_conv = wR2(numClasses)
    model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv.load_state_dict(torch.load(resume_file))
    model_conv = model_conv.cuda()
else:
    model_conv = wR2(numClasses)
    if use_gpu:
        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv = model_conv.cuda()

print(model_conv)
print(get_n_params(model_conv))


criterion = nn.MSELoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=120, gamma=0.1)

# optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.01)

# dst = LocDataLoader([args["images"]], imgSize)
dst = ChaLocDataLoader(args["images"].split(','), imgSize)
trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=0)


def train_model(model, criterion, optimizer, num_epochs=25):
    # since = time.time()
    for epoch in range(epoch_start, num_epochs):
        lossAver = []
        model.train(True)
        lrScheduler.step()
        start = time()

        for i, (XI, YI) in enumerate(trainloader):
            # print('%s/%s %s' % (i, times, time()-start))
            YI = np.array([el.numpy() for el in YI]).T
            if use_gpu:
                x = Variable(XI.cuda(0))
                y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
            else:
                x = Variable(XI)
                y = Variable(torch.FloatTensor(YI), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)
            #print(y_pred.cpu().detach().numpy())
            #print(y_pred.shape)

            # Compute and print loss
            loss = 0.0
            if len(y_pred) == batchSize:
                loss +=  nn.L1Loss().cuda()(y_pred[:,:4], y[:,:4])
                loss +=  nn.L1Loss().cuda()(y_pred[:,4:], y[:,4:])
                lossAver.append(loss.item())

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.save(model.state_dict(), storeName)
            if i % 200 == 1:
                with open(args['writeFile'], 'a') as outF:
                    outF.write('train %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver[-50:]) / len(lossAver[-50:])))
                print('train %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver[-50:]) / len(lossAver[-50:])))
        print ('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        with open(args['writeFile'], 'a') as outF:
            outF.write('Epoch: %s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        if epoch%10==1:
            torch.save(model.state_dict(), storeName + str(epoch))
    return model


model_conv = train_model(model_conv, criterion, optimizer_conv, num_epochs=epochs)
