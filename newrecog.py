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
ap.add_argument("-w", "--writeFile", default='RECOG.out',
                help="file for output")
ap.add_argument("-t", "--test", required=True,
                help="dirs for test")
args = vars(ap.parse_args())

use_gpu = torch.cuda.is_available()
print (use_gpu)
provNum, alphaNum, adNum = 38, 25, 35
numClasses = 8
imgSize = (50, 50)
batchSize = int(args["batchsize"]) if use_gpu else 8
modelFolder = 'recog/'
trainDirs = args["images"].split(',')
testDirs = args["test"].split(',')
storeName = modelFolder + 'newrecog.pth'
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


class Recog(nn.Module):
    def __init__(self, num_classes=1000):
        super(Recog, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )

        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )

        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
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
        self.classifier1 = nn.Sequential(
            nn.Linear(10368, 128),
            # nn.ReLU(inplace=True),

            # nn.ReLU(inplace=True),
            nn.Linear(128, provNum),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(10368, 128),
            # nn.ReLU(inplace=True),

            # nn.ReLU(inplace=True),
            nn.Linear(128, alphaNum),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(10368, 128),
            # nn.ReLU(inplace=True),

            # nn.ReLU(inplace=True),
            nn.Linear(128, adNum),
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(10368, 128),
            # nn.ReLU(inplace=True),

            # nn.ReLU(inplace=True),
            nn.Linear(128, adNum),
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(10368, 128),
            # nn.ReLU(inplace=True),

            # nn.ReLU(inplace=True),
            nn.Linear(128, adNum),
        )
        self.classifier6 = nn.Sequential(
            nn.Linear(10368, 128),
            # nn.ReLU(inplace=True),

            # nn.ReLU(inplace=True),
            nn.Linear(128, adNum),
        )
        self.classifier7 = nn.Sequential(
            nn.Linear(10368, 128),
            # nn.ReLU(inplace=True),

            # nn.ReLU(inplace=True),
            nn.Linear(128, adNum),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        y0 = self.classifier1(x11)
        y1 = self.classifier2(x11)
        y2 = self.classifier3(x11)
        y3 = self.classifier4(x11)
        y4 = self.classifier5(x11)
        y5 = self.classifier6(x11)
        y6 = self.classifier7(x11)
        return [y0,y1,y2,y3,y4,y5,y6]
def isEqual(labelGT, labelP):
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    # print(sum(compare))
    return sum(compare)

def eval(model,testDirs):
    count, error, correct = 0, 0, 0
    dst = labelTestDataLoader(testDirs, imgSize)
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

        y_pred = model(x)

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
    model_conv = Recog(numClasses)
    model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
    model_conv.load_state_dict(torch.load(resume_file))
    model_conv = model_conv.cuda()
else:
    model_conv = Recog(numClasses)
    if use_gpu:
        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv = model_conv.cuda()

print(model_conv)
print(get_n_params(model_conv))


criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=150, gamma=0.1)

# optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.01)

# dst = LocDataLoader([args["images"]], imgSize)
dst = labelFpsDataLoader(trainDirs, imgSize)
trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=0)


def train_model(model, criterion, optimizer, num_epochs=25):
    # since = time.time()
    for epoch in range(epoch_start, num_epochs):
        lossAver = []
        model.train(True)
        lrScheduler.step()
        start = time()

        for i, (XI, labels, ims) in enumerate(trainloader):
            if not len(XI) == batchSize:
                print(len(XI))
                continue

            YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]

            if use_gpu:
                x = Variable(XI.cuda(0))

            else:
                x = Variable(XI)

            # Forward pass: Compute predicted y by passing x to the model

            try:
                y_pred = model(x)
                # print(y_pred.shape)

            except:
                print('2')
                continue

            # Compute and print loss
            loss = 0.0
            for j in range(7):
                # print(YI)
                l = Variable(torch.LongTensor([el[j] for el in YI]).cuda(0))
                loss += criterion(y_pred[j], l)

                # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), storeName)
            try:
                lossAver.append(loss.item())
                #print(len(lossAver))

            except:

                print('errorr')
                pass
            if i % 200 == 1:
                with open(args['writeFile'], 'a') as outF:
                    outF.write('train %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver[-50:]) / len(lossAver[-50:])))
                print('train %s images, use %s seconds, loss %s\n' % (i*batchSize, time() - start, sum(lossAver[-50:]) / len(lossAver[-50:])))
        print ('%s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        model.eval()
        count, correct, error, precision, avgTime = eval(model, testDirs)
        #count1, correct1, error1, precision1, avgTime1 = eval(model, trainDirs)
        print(precision)
        with open(args['writeFile'], 'a') as outF:
            outF.write('Epoch: %s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), time()-start))
        if epoch%10==1:
            torch.save(model.state_dict(), storeName + str(epoch))
    return model


model_conv = train_model(model_conv, criterion, optimizer_conv, num_epochs=epochs)
