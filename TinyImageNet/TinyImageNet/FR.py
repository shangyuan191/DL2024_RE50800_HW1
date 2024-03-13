#!/usr/bin/env python
# coding: utf-8


import os
from tqdm import tqdm
dd=os.listdir('TIN')
f1 = open('train.txt', 'w')
f2 = open('test.txt', 'w')


# 使用 tqdm 函式包裝迭代對象
test_ratio=0.2
for i in tqdm(range(len(dd)), desc='Processing folders', leave=True):  #len(dd)=200 (num of folder)
    d2 = os.listdir(f"TIN/{dd[i]}/images/")          # len(d2)=500 (num of image per folder)
    train_num=round(len(d2)*(1-test_ratio))
    test_num=round(len(d2)*test_ratio)
    for j in tqdm(range(train_num), desc=f'Processing images in folder {i}', leave=False):
        str1 = f"TIN/{dd[i]}/images/{d2[j]}"
        f1.write(f"{str1} {i}\n")
    for j in tqdm(range(train_num,len(d2)), desc=f'Processing images in folder {i}', leave=False):
        str1 = f"TIN/{dd[i]}/images/{d2[j]}"
        f2.write(f"{str1} {i}\n")



f1.close()
f2.close()



import numpy as np
from numpy import linalg as LA
import cv2

def load_img(f):
    f=open(f)
    lines=f.readlines()
    imgs, labels=[], []
    for i in tqdm(range(len(lines)), desc='Loading images', leave=True):
        fn, label = lines[i].split(' ')
        #print(fn)
        
        im1=cv2.imread(fn)
        im1=cv2.resize(im1, (256,256))   ## convert (64,64,3) to (256,256,3)
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) ## convert (256,256,3) to (256,256)        
        '''===============================
        影像處理的技巧可以放這邊，來增強影像的品質
        
        ==============================='''
        
        vec = np.reshape(im1, [-1])
        imgs.append(vec) 
        labels.append(int(label))


    imgs= np.asarray(imgs,np.float32)
    labels= np.asarray(labels, np.int32)
    return imgs, labels 


x_train, y_train = load_img('train.txt')
x_test, y_test = load_img('test.txt')
print(x_train,y_train,x_test,y_test)



#======================================
#X就是資料，Y是Label，請設計不同分類器來得到最高效能
#必須要計算出分類的正確率
#======================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score,accuracy_score

knn = KNeighborsClassifier(n_neighbors=5)

Kfold = KFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
accuracies = []

for fold, (train_index, val_index) in enumerate(tqdm(Kfold.split(x_train), desc='Cross-validation', total=Kfold.n_splits)):
    x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

    # 將整個訓練集載入模型
    knn.fit(x_fold_train, y_fold_train)

    y_pred = knn.predict(x_fold_val)

    f1 = f1_score(y_fold_val, y_pred, average='macro')
    acc = accuracy_score(y_fold_val, y_pred)

    f1_scores.append(f1)
    accuracies.append(acc)

print(f"Average F1-score:{np.mean(f1_scores)}")
print(f"Average Accuracy:{np.mean(accuracies)}")

from sklearn.svm import SVC
from sklearn.decomposition import PCA
# 創建PCA實例，指定要降到的維度
pca = PCA(n_components=100)  # 假設降到100維

# 在訓練數據上擬合PCA模型並轉換數據
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# 創建SVM分類器
svm = SVC(kernel='linear', random_state=42)

Kfold = KFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []
accuracies = []

for fold, (train_index, val_index) in enumerate(tqdm(Kfold.split(x_train_pca), desc='Cross-validation', total=Kfold.n_splits)):
    x_fold_train, x_fold_val = x_train_pca[train_index], x_train_pca[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

    # 將整個訓練集載入模型
    svm.fit(x_fold_train, y_fold_train)

    y_pred = svm.predict(x_fold_val)

    f1 = f1_score(y_fold_val, y_pred, average='macro')
    acc = accuracy_score(y_fold_val, y_pred)

    f1_scores.append(f1)
    accuracies.append(acc)

print(f"Average F1-score:{np.mean(f1_scores)}")
print(f"Average Accuracy:{np.mean(accuracies)}")