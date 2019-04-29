from torch.utils.data import *
from imutils import paths
import cv2
import numpy as np
import math
import os
minPlateRatio = 1  # 车牌比例
maxPlateRatio = 6

lower_blue = np.array([100, 40, 46])
higher_blue = np.array([124, 255, 255])
lower_blue1 = np.array([100, 40, 50])
higher_blue1 = np.array([140, 255, 255])
# 找到符合车牌形状的矩形
def findPlateNumberRegion(img):
    region = []
    xcoorpool=[]
    ycoorpool=[]
    lengthpool=[]
    objh,objw=img.shape
    contours_img, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print("contours lenth is :%s" % (len(contours)))
    list_rate = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < 1000: #(objh*objw/80):
            continue
        rect = cv2.minAreaRect(cnt)
        # print("rect is:%s" % {rect})
        box = np.int32(cv2.boxPoints(rect))
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        ratio = float(width) / float(height)
        xcoor,ycoor,lengthx = getxyRate(cnt)
        #print("area", area, "ratio:", ratio)
        if ratio > maxPlateRatio or ratio < minPlateRatio:
            continue
        xcoorpool.append(xcoor)
        ycoorpool.append(ycoor)
        lengthpool.append(lengthx)
        list_rate.append(ratio)
    index = getSatifyestBox(list_rate)
    return xcoorpool[index],ycoorpool[index],lengthpool[index]

def getSatifyestBox(list_rate):
    for index, key in enumerate(list_rate):
        list_rate[index] = abs(key - 3)
    #print(list_rate)
    index = list_rate.index(min(list_rate))
    #print(index)
    return index


def getxyRate(cnt):

    x_list = []
    y_list = []
    for location_value in cnt:
        location = location_value[0]
        x_list.append(location[0])
        y_list.append(location[1])
    x_height = max(x_list) - min(x_list)
    xcoordinate=(max(x_list) + min(x_list))/2

    ycoordinate=(max(y_list) + min(y_list))/2

    return xcoordinate,ycoordinate,x_height


def prelocation(file):
    img = cv2.imread(file)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_blue, higher_blue)
    res = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    sobel = cv2.convertScaleAbs(cv2.Sobel(gaussian, cv2.CV_16S, 1, 0, ksize=3))
    ret, binary = cv2.threshold(sobel, 150, 255, cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    xmean,ymean ,lengthpred = findPlateNumberRegion(closed)

    return xmean,ymean,lengthpred

class labelFpsDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        # img = img.astype('float32')
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2,0,1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]



        return resizedImage, lbl, img_name


class labelTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        #print(img_name)
        img = cv2.imread(img_name)

        # img = img.astype('float32')
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2,0,1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        lbl = img_name.split('/')[-1].split('.')[0].split('-')[-3]

        return resizedImage, lbl, img_name



class ChaLocDataLoader(Dataset):
    def __init__(self, img_dir,imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_blue1, higher_blue1)
        img = cv2.bitwise_and(img, img, mask=mask)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.reshape(resizedImage, (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]))

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        [rightdown,leftdown, leftup,rightup] = [[int(eel) for eel in el.split('&')] for el in iname[3].split('_')]

        # tps = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        # for dot in tps:
        #     cv2.circle(img, (int(dot[0]), int(dot[1])), 2, (0, 0, 255), 2)
        # cv2.imwrite("/home/xubb/1_new.jpg", img)

        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        assert img.shape[0] == 1160
        new_labels = [rightdown[0]/ori_w,leftdown[0]/ori_w,leftup[0]/ori_w,rightup[0]/ori_w,rightdown[1]/ori_h,leftdown[1]/ori_h,leftup[1]/ori_h,rightup[1]/ori_h]

        resizedImage = resizedImage.astype('float32')
        # Y = Y.astype('int8')
        resizedImage /= 255.0
        # lbl = img_name.split('.')[0].rsplit('-',1)[-1].split('_')[:-1]
        # lbl = img_name.split('/')[-1].split('.')[0].rsplit('-',1)[-1]
        # lbl = map(int, lbl)
        # lbl2 = [[el] for el in lbl]

        # resizedImage = torch.from_numpy(resizedImage).float()
        return resizedImage, new_labels


class demoTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        stayori=0
        img = cv2.imread(img_name)
        realh, realw, imvalue = img.shape
        predw = math.floor(0.621 * realh)
        predh = math.floor(1.611 * realw)
        addh = math.floor((predh - realh) / 2)
        addw = math.floor((predw - realw) / 2)
        try:
            xmean, ymean, lengthpred=prelocation(img_name)
        except:
            stayori=1
            print('huge error')
        if stayori==1:
            if (realw / realh) < 0.6:
                img = cv2.copyMakeBorder(img, 0, 0, addw, addw, cv2.BORDER_REPLICATE)
            elif (realw / realh) > 0.64 :
                img = cv2.copyMakeBorder(img, addh, addh, 0, 0, cv2.BORDER_REPLICATE)
        realh, realw, imvalue = img.shape
        if (realw / realh) > 0.64:
            if realh * 0.62 * 0.6 < lengthpred:
                img = cv2.copyMakeBorder(img, addh, addh, 0, 0, cv2.BORDER_REPLICATE)
                print('error')
            else:
                left_pred = math.floor(xmean - predw / 2)
                right_pred = math.floor(xmean + predw/ 2)
                if right_pred>realw :
                    left_pred+=right_pred-realw
                    right_pred=realw
                elif left_pred<0 :
                    right_pred=right_pred-left_pred
                    left_pred=0
                img = img[:, left_pred:right_pred]
                #cv2.imshow('image1', img)
                #cv2.waitKey(0)
                #cv2.imwrite('car_id/1.jpg', img)
        elif (realw / realh) < 0.6:
            up_pred = math.floor(ymean + predh / 2)
            down_pred = math.floor(ymean - predh / 2)
            if up_pred>realh:
                down_pred+=up_pred-realh
                up_pred=realh
            elif down_pred<0 :
                up_pred=up_pred-down_pred
                down_pred=0
            img = img[down_pred:up_pred, :]
            #cv2.imwrite('car_id/0.jpg', img)
        #img = new_img
        cv2.imwrite(img_name, img)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_blue1, higher_blue1)
        img = cv2.bitwise_and(img, img, mask=mask)
        # img = img.astype('float32')
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.reshape(resizedImage, (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        return resizedImage, img_name
class ChaLocDataLoader1(Dataset):
    def __init__(self, img_dir,imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_blue1, higher_blue1)
        img = cv2.bitwise_and(img, img, mask=mask)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.reshape(resizedImage, (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]))

        #iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        #[rightdown,leftdown, leftup,rightup] = [[int(eel) for eel in el.split('&')] for el in iname[3].split('_')]

        # tps = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        # for dot in tps:
        #     cv2.circle(img, (int(dot[0]), int(dot[1])), 2, (0, 0, 255), 2)
        # cv2.imwrite("/home/xubb/1_new.jpg", img)

        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        #assert img.shape[0] == 1160
        #new_labels = [rightdown[0]/ori_w,leftdown[0]/ori_w,leftup[0]/ori_w,rightup[0]/ori_w,rightdown[1]/ori_h,leftdown[1]/ori_h,leftup[1]/ori_h,rightup[1]/ori_h]
        new_labels =0
        resizedImage = resizedImage.astype('float32')
        # Y = Y.astype('int8')
        resizedImage /= 255.0
        # lbl = img_name.split('.')[0].rsplit('-',1)[-1].split('_')[:-1]
        # lbl = img_name.split('/')[-1].split('.')[0].rsplit('-',1)[-1]
        # lbl = map(int, lbl)
        # lbl2 = [[el] for el in lbl]

        # resizedImage = torch.from_numpy(resizedImage).float()
        return resizedImage, new_labels ,img_name







