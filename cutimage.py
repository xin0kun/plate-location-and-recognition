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
import math



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", default='C:/Users/54186/rpnet/demo',
                help="path to the input file")
#ap.add_argument("-n", "--epochs", default=25,
                #help="epochs for train")
#ap.add_argument("-b", "--batchsize", default=4,
                #help="batch size for train")
ap.add_argument("-r", "--resume", required=True,
                help="file for re-train")
#ap.add_argument("-w", "--writeFile", default='wR2.out',
                #help="file for output")
args = vars(ap.parse_args())

use_gpu = torch.cuda.is_available()
print (use_gpu)

numClasses = 8
numPoints = 4
imgSize = (480, 480)
batchSize = 8 if use_gpu else 8
#if not os.path.isdir(modelFolder):
  #  os.mkdir(modelFolder)

#epochs = int(args["epochs"])
#   initialize the output file
#with open(args['writeFile'], 'wb') as outF:
 #   pass


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


epoch_start = 0
resume_file = str(args["resume"])

model_conv = wR2(numClasses)
model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
model_conv.load_state_dict(torch.load(resume_file))
model_conv = model_conv.cuda()
model_conv.eval()


#print(model_conv)
print(get_n_params(model_conv))


#criterion = nn.MSELoss()
#optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
#lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=120, gamma=0.1)

# optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.01)

# dst = LocDataLoader([args["images"]], imgSize)

dst = ChaLocDataLoader1(args["images"].split(','), imgSize)
trainloader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=0)
img_paths=[]
img_dir=args["images"].split(',')
start = time()

for i, (XI, lash,ims) in enumerate(trainloader):
    print(i)
    if use_gpu:
        x = Variable(XI.cuda(0))
    else:
        x = Variable(XI)
    # Forward pass: Compute predicted y by passing x to the model

    fps_pred = model_conv(x)
    print(ims[0])

    filename = ims[0].split('/')[-1].split('\\')[-1]
    #outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
    #labelPred = [t[0].index(max(t[0])) for t in outputY]
    img = cv2.imread(ims[0])
    ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
    [xx0,xx1,xx2,xx3,yy0,yy1,yy2,yy3] = fps_pred.data.cpu().numpy()[0].tolist()
    print(xx0,xx1,xx2,xx3,yy0,yy1,yy2,yy3)
    points=[(int(xx0*ori_w),int(yy0*ori_h)),(int(xx1*ori_w),int(yy1*ori_h)),(int(xx2*ori_w),int(yy2*ori_h)),(int(xx3*ori_w),int(yy3*ori_h))]
    for point in points:
        cv2.circle(img, point, 1, (0, 0, 255), 4)
    cv2.imwrite(ims[0], img)
    labelxmax=max(xx0,xx1,xx2,xx3)
    labelxmin=min(xx0,xx1,xx2,xx3)
    labelymax=max(yy0,yy1,yy2,yy3)
    labelymin=min(yy0,yy1,yy2,yy3)
    img=img[int(labelymin*ori_h):int(labelymax*ori_h),int(labelxmin*ori_w):int(labelxmax*ori_w)]
    #cv2.imwrite('C:/Users/54186/rpnet/demo1/%s' %filename,img)

    #img = cv2.imread(ims[0])
    #left_up = [(cx - w/2)*img.shape[1], (cy - h/2)*img.shape[0]]
    #print(left_up)
    #right_down = [(cx + w/2)*img.shape[1], (cy + h/2)*img.shape[0]]
    #print(right_down)
    #img_new=img[int(left_up[1]):int(right_down[1]),int(left_up[0]):int(right_down[0])]
    #cv2.imwrite('C:/Users/54186/rpnet/234/%s' %filename, img_new)
    #cv2.rectangle(img, (int(left_up[0]), int(left_up[1])), (int(right_down[0]), int(right_down[1])), (0, 0, 255), 2)
    #   The first character is Chinese character, can not be printed normally, thus is omitted.
    #lpn = alphabets[labelPred[1]] + ads[labelPred[2]] + ads[labelPred[3]] + ads[labelPred[4]] + ads[labelPred[5]] + ads[labelPred[6]]
    #cv2.putText(img, lpn, (int(left_up[0]), int(left_up[1])-20), cv2.FONT_ITALIC, 2, (0, 0, 255))
    #cv2.imwrite(ims[0], img)
    #lpn =provinces[labelPred[0]]+ alphabets[labelPred[1]] + ads[labelPred[2]] + ads[labelPred[3]] + ads[labelPred[4]] + ads[labelPred[5]] + ads[labelPred[6]]
    #print('%s' %lpn)

