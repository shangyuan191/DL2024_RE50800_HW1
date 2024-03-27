#!/usr/bin/env python
# coding: utf-8


import os
from tqdm import tqdm
import numpy as np
from numpy import linalg as LA
import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import time
def HOG(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    window_size=img.shape
    block_size=(16,16)
    block_stride=(8,8)
    cell_size=(8,8)
    n_bins=9

    hog=cv2.HOGDescriptor(window_size,block_size,block_stride,cell_size,n_bins)
    features=hog.compute(img).flatten()
    return features

def ColorHistogram(img,bins=(8,8,8)):
    HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([HSV],[0,1,2],None,bins,[0, 180, 0, 256, 0, 256])
    hist=cv2.normalize(hist,hist).flatten()
    return hist

def SIFT(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift=cv2.SIFT_create()
    keypoints,descriptors=sift.detectAndCompute(img,None)
    return keypoints,descriptors


def build_vocabulary(descriptors,k=100):
    descriptors_array=np.vstack(descriptors)
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(descriptors_array)
    return kmeans

def bow_encoding(descriptors,visual_words):
    visual_words.predict(descriptors)
    histogram=np.bincount(visual_words.labels_,minlength=visual_words.n_clusters)
    return histogram

def train(x_train,y_train,model_name):
    if model_name=="KNN":
        model=KNeighborsClassifier(n_neighbors=20)
        model.fit(x_train,y_train)
        return model

    elif model_name=="SVM":
        model=SVC(kernel='rbf',C=1.0,gamma="auto")
        model.fit(x_train,y_train)
        return model
    elif model_name=="RandomForest":
        model=RandomForestClassifier(n_estimators=100,random_state=42)
        model.fit(x_train,y_train)
        return model


def test(x_test,y_test,model):
    y_predict=model.predict(x_test)
    acc=accuracy_score(y_test,y_predict)
    F1score=f1_score(y_test,y_predict,average='macro')
    classification=classification_report(y_test,y_predict)
    print(f"Acc: {acc}")
    print(f"F1 score: {F1score}")
    print(f"Classification Report: {classification}")
    return acc,F1score,classification

def TrainTestSplit():
    print("Start train test split...")
    dd=os.listdir('TinyImageNet/TIN')
    f1 = open('train.txt', 'w')
    f2 = open('test.txt', 'w')


    # 使用 tqdm 函式包裝迭代對象
    test_ratio=0.2
    for i in tqdm(range(len(dd)//2), desc='Processing folders', leave=True):  #len(dd)=200 (num of folder)
        d2 = os.listdir(f"TinyImageNet/TIN/{dd[i]}/images/")          # len(d2)=500 (num of image per folder)
        train_num=round(len(d2)*(1-test_ratio))
        test_num=round(len(d2)*test_ratio)
        for j in tqdm(range(train_num), desc=f'Processing images in folder {i}', leave=False):
            str1 = f"TinyImageNet/TIN/{dd[i]}/images/{d2[j]}"
            f1.write(f"{str1} {i}\n")
        for j in tqdm(range(train_num,len(d2)), desc=f'Processing images in folder {i}', leave=False):
            str1 = f"TinyImageNet/TIN/{dd[i]}/images/{d2[j]}"
            f2.write(f"{str1} {i}\n")



    f1.close()
    f2.close()
    print("Train test split done!")



def load_img(f,method):
    f=open(f)
    lines=f.readlines()
    imgs, labels=[], []
    all_descriptors=[]
    if method=="SIFT":
        for i in tqdm(range(len(lines)),desc='SIFT processing...',leave=True):
            fn, label = lines[i].split(' ')
            im1=cv2.imread(fn)
            im1=cv2.resize(im1,(32,32))
            _,descriptors=SIFT(im1)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        vw=build_vocabulary(all_descriptors)
    for i in tqdm(range(len(lines)), desc='Loading images', leave=True):
        fn, label = lines[i].split(' ')
        #print(fn)
        
        im1=cv2.imread(fn)
        im1=cv2.resize(im1, (32,32))   ## convert (64,64,3) to (32,32,3)
        #影像處理的技巧可以放這邊，來增強影像的品質
        if method=="HOG":
            im1=HOG(im1)
        elif method=="ColorHistogram":
            im1=ColorHistogram(im1)
        elif method=="SIFT":
            kp,im1=SIFT(im1)
            if im1 is not None:
                im1=bow_encoding(im1,vw)

        if im1 is not None: 
            vec = np.reshape(im1, [-1])
            imgs.append(vec) 
            labels.append(int(label))


    imgs= np.asarray(imgs,np.float32)
    labels= np.asarray(labels, np.int32)
    return imgs, labels 



if __name__ == "__main__":
    TrainTestSplit()
    print("\n\n")
    for FEM in ["HOG","ColorHistogram","SIFT"]:
        method_time_start=time.time()
        print(f"{FEM} method start...")
        x_train,y_train=load_img("train.txt",FEM)
        x_test,y_test=load_img("test.txt",FEM)
        print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
        method_time_end=time.time()
        print(f"{FEM} method end,spend {round(method_time_end-method_time_start,2)} sec.\n\n")
        for md in ["KNN","SVM","RandomForest"]:
            model_time_start=time.time()
            print(f"Now model is {md},model start...")
            model=train(x_train,y_train,md)
            acc,F1score,classification=test(x_test,y_test,model)
            model_time_end=time.time()
            print(f"{md} model end,spend {round(model_time_end-model_time_start,2)} sec.\n")

            with open('record.txt','a') as f:
                f.write(f"(Method , Model) : ({FEM} , {md})\n")
                f.write(f"Method spend {round(method_time_end-method_time_start,2)} sec.\n")
                f.write(f"Model spend {round(model_time_end-model_time_start,2)} sec.\n")
                f.write(f"Accuracy = {acc}\n")
                f.write(f"F1 score = {F1score}\n")
                f.write(f"Classification Report: {classification}\n\n")


            
            



# #======================================
# #X就是資料，Y是Label，請設計不同分類器來得到最高效能
# #必須要計算出分類的正確率
# #======================================
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import KFold
# from sklearn.metrics import f1_score,accuracy_score

# knn = KNeighborsClassifier(n_neighbors=5)

# Kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# f1_scores = []
# accuracies = []

# for fold, (train_index, val_index) in enumerate(tqdm(Kfold.split(x_train), desc='Cross-validation', total=Kfold.n_splits)):
#     x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
#     y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

#     # 將整個訓練集載入模型
#     knn.fit(x_fold_train, y_fold_train)

#     y_pred = knn.predict(x_fold_val)

#     f1 = f1_score(y_fold_val, y_pred, average='macro')
#     acc = accuracy_score(y_fold_val, y_pred)

#     f1_scores.append(f1)
#     accuracies.append(acc)

# print(f"Average F1-score:{np.mean(f1_scores)}")
# print(f"Average Accuracy:{np.mean(accuracies)}")

# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# # 創建PCA實例，指定要降到的維度
# pca = PCA(n_components=100)  # 假設降到100維

# # 在訓練數據上擬合PCA模型並轉換數據
# x_train_pca = pca.fit_transform(x_train)
# x_test_pca = pca.transform(x_test)

# # 創建SVM分類器
# svm = SVC(kernel='linear', random_state=42)

# Kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# f1_scores = []
# accuracies = []

# for fold, (train_index, val_index) in enumerate(tqdm(Kfold.split(x_train_pca), desc='Cross-validation', total=Kfold.n_splits)):
#     x_fold_train, x_fold_val = x_train_pca[train_index], x_train_pca[val_index]
#     y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

#     # 將整個訓練集載入模型
#     svm.fit(x_fold_train, y_fold_train)

#     y_pred = svm.predict(x_fold_val)

#     f1 = f1_score(y_fold_val, y_pred, average='macro')
#     acc = accuracy_score(y_fold_val, y_pred)

#     f1_scores.append(f1)
#     accuracies.append(acc)

# print(f"Average F1-score:{np.mean(f1_scores)}")
# print(f"Average Accuracy:{np.mean(accuracies)}")