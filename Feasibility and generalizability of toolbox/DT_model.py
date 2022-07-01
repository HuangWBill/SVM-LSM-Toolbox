# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import arcpy
arcpy.env.overwriteOutput = True

v=sklearn.__version__
if int(v[2:4])<=20:
    from sklearn.externals import joblib
else:
    import joblib

def evaluate_index(labels, result,pred):
    con_mat = confusion_matrix(labels, result)
    accuracy = metrics.accuracy_score(labels, result)
    precision = metrics.precision_score(labels, result)
    recall = metrics.recall_score(labels, result)
    f1 = metrics.f1_score(labels, result)
    RMSE = (metrics.mean_squared_error(labels, result)) ** 0.5
    kappa = metrics.cohen_kappa_score(labels, result)
    fpr, tpr, threshold = roc_curve(labels, pred)
    roc_auc = auc(fpr, tpr)
    return con_mat,accuracy,precision,recall,f1,RMSE,kappa,roc_auc,fpr, tpr

def DT(data_root_path,model_save,test_result_txt,train_result_txt,row_patch_size,col_patch_size,channel,max_depth,min_samples_leaf):
    test_file_path = data_root_path + "test.txt"
    train_file_path = data_root_path + "train.txt"

    def train_mapper(sample):
        img = sample
        if not os.path.exists(img):
            arcpy.AddMessage(img+"haven't fig")
        img = arcpy.Raster(img)
        img=arcpy.RasterToNumPyArray(img)
        img = img.reshape((1, img.shape[0] * img.shape[1] * img.shape[2]), order='C')
        return img

    def train_r(train_list):
        dataLabel = []
        i = 0
        with open(train_list, "r") as f:
            lines = [line.strip() for line in f]
            dataNum = len(lines)
            dataMat = np.zeros((dataNum, int(channel) * int(row_patch_size) * int(col_patch_size)))
            for line in lines:
                img_path, lab = line.replace("\n", "").split("\t")
                dataLabel.append(int(lab))
                dataMat[i, :] = train_mapper(img_path)
                i = i + 1
        return dataMat, dataLabel

    def test_mapper(sample):
        img = sample
        if not os.path.exists(img):
            arcpy.AddMessage(img+"haven't fig")
        img = arcpy.Raster(img)
        img=arcpy.RasterToNumPyArray(img)
        img = img.reshape((1, img.shape[0] * img.shape[1] * img.shape[2]), order='C')
        return img

    train_image, train_label = train_r(train_file_path)
    with open(test_result_txt, "w") as f:
        pass
    with open(train_result_txt, "w") as f:
        pass

    predictor = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    predictor.fit(train_image, train_label)
    joblib.dump(predictor,model_save+'\DT_md_'+str(max_depth)+'_msl_'+str(min_samples_leaf)+'.model')

    result1 = []
    labels1 = []

    errorCount1 = 0.0
    with open(train_file_path, "r") as f:
        lines = [line.strip() for line in f]
        dataNum1 = len(lines)
        for line in lines:
            i = 0
            img_path, train_dataLabel = line.replace("\n", "").split("\t")
            train_dataMat = train_mapper(img_path)
            classifierResult1 = predictor.predict(train_dataMat)
            probability1 = predictor.predict_proba(train_dataMat)
            result1.append(probability1[0][1])
            labels1.append(int(train_dataLabel))
            with open(train_result_txt, "a") as f:
                line = "%s \t %d \t %f\n" % (img_path, int(train_dataLabel), probability1[0][1])
                f.write(line)
            if (classifierResult1[0] != int(train_dataLabel)):
                errorCount1 += 1.0
    errorCount = 0.0
    result = []
    labels = []
    pred=[]
    with open(test_file_path, "r") as f:
        lines = [line.strip() for line in f]
        dataNum = len(lines)
        for line in lines:
            img_path, test_dataLabel = line.replace("\n", "").split("\t")
            test_dataMat = test_mapper(img_path)
            classifierResult = predictor.predict(test_dataMat)
            probability = predictor.predict_proba(test_dataMat)
            pred.append(probability[0][1])
            labels.append(int(test_dataLabel))
            result.append(classifierResult[0])
            with open(test_result_txt, "a") as f:
                line = "%s \t %d \t %f\t %d\n" % (img_path, int(test_dataLabel), probability[0][1], classifierResult[0])
                f.write(line)
            if (classifierResult[0] != int(test_dataLabel)):
                errorCount += 1.0
        arcpy.AddMessage("testset predict error number:\t%d\n accuracy:%f" % (errorCount, (dataNum - errorCount) / dataNum))
        test_acc = (dataNum - errorCount) / dataNum
    arcpy.AddMessage("trainset predict error number:\t%d\n accuracy:%f" % (errorCount1, (dataNum1 - errorCount1) / dataNum1))
    train_acc = (dataNum1 - errorCount1) / dataNum1
    con_mat, accuracy, precision, recall, f1, RMSE, kappa, roc_auc, fpr, tpr = evaluate_index(labels, result,pred)
    evaluate_txt=model_save+'/evaluate_result.txt'
    with open(evaluate_txt, "w") as f:
        pass
    with open(evaluate_txt, "a") as f:
        line = "confusion_matrix: \n" +str(con_mat)+"\naccuracy:"+ str(accuracy)+ "\nprecision:"+ str(precision)+"\nrecall:"+ str(recall)+"\nf1:"+ str(f1)+"\nroc_auc:"+ str(roc_auc)
        f.write(line)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    lw = 3
    plt.figure(0)
    plt.plot(fpr, tpr, lw=lw, label='DT:AUC=%0.4f' % roc_auc)
    plt.rcParams['savefig.dpi'] = 350
    plt.rcParams['figure.dpi'] = 350
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.tick_params(labelsize=18)
    plt.ylim([0.0, 1.0])
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.xlabel('FPR', fontsize=25,fontproperties='Times New Roman')
    plt.ylabel('TPR', fontsize=25,fontproperties='Times New Roman')
    plt.legend(loc="lower right", handlelength=4, fontsize=18)
    plt.savefig(model_save+'\ROC.png', bbox_inches='tight')
    plt.close()
    return roc_auc,test_acc,train_acc

data_root_path = arcpy.GetParameterAsText(0)
model_save_path=arcpy.GetParameterAsText(1)
result_txt = model_save_path+'/parameter_result_txt.txt'
png_path=model_save_path+'/parameter_result_png.png'
svg_path=model_save_path+'/parameter_result_png.svg'
max_depth = arcpy.GetParameterAsText(2)
min_samples_leaf = arcpy.GetParameterAsText(3)
max_depth = eval(max_depth)
min_samples_leaf = eval(min_samples_leaf)
row_patch_size = arcpy.GetParameterAsText(4)
col_patch_size = arcpy.GetParameterAsText(5)
channel = arcpy.GetParameterAsText(6)

AUC = []
test_acc, train_acc = [], []
with open(result_txt, "w") as f:
    pass
with open(result_txt, "a") as f:
    line = "max_depth\tmin_samples_leaf\tAUC\tTest_acc\tTrain_acc\tTrain_acc-Test_acc\n"
    f.write(line)

arcpy.AddMessage("******************************begain train!******************************")
arcpy.AddMessage("criterion=gini")
for i in max_depth:
    for j in min_samples_leaf:
        model_save = model_save_path+'/max_depth_'+str(i)+'_min_samples_leaf_'+str(j)
        if not os.path.exists(model_save):
            os.makedirs(model_save)
        train_result_txt = model_save+'/DT_train_result_txt.txt'
        test_result_txt = model_save+'/DT_test_result_txt.txt'
        roc_auc, testacc, trainacc = DT(data_root_path, model_save,test_result_txt, train_result_txt, row_patch_size,col_patch_size, channel, i, j)
        with open(result_txt, "a") as f:
            line = "%f\t%f\t%f\t%f\t%f\t%f\n" % (i, j, roc_auc, testacc, trainacc,trainacc-testacc)
            f.write(line)
        arcpy.AddMessage("********************max_depth=" + str(i) + "," + "min_samples_leaf=" + str(j) + "DT train End!********************")

plt.rcParams['savefig.dpi'] = 350
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
result_df = pd.read_csv(result_txt, sep='\t', index_col=False)
fig=plt.scatter(x=result_df['max_depth'],y=result_df['min_samples_leaf'],s=((result_df['AUC']-min(result_df['AUC'])))/(max(result_df['AUC'])-min(result_df['AUC']))*100,c=result_df['Train_acc-Test_acc'],cmap='rainbow',vmin=0, vmax=0.5)
cb=plt.colorbar(fig)
plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0)
plt.ylabel("min_samples_leaf", fontproperties='Times New Roman',fontsize=10)
plt.xlabel("max_depth", fontproperties='Times New Roman',fontsize=10)
plt.savefig(png_path, bbox_inches='tight')
plt.savefig(svg_path, bbox_inches='tight')
plt.show()
arcpy.AddMessage("*************************All parameter values training completed!*************************")

