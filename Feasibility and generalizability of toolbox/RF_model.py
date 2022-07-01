# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
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

def RF(data_root_path,model_save,test_result_txt,train_result_txt,row_patch_size,col_patch_size,channel,max_depth,max_features,n_estimators,min_samples_leaf,min_samples_split):
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

    predictor = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features, min_samples_leaf=min_samples_leaf)

    predictor.fit(train_image, train_label)
    joblib.dump(predictor,model_save+'\RF_md_'+str(max_depth)+'_mf_' + str(max_features) + '_n_' + str(n_estimators) + '_msl_' + str(min_samples_leaf) + '_mss_' + str(min_samples_split)+'.model')

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
    plt.plot(fpr, tpr, lw=lw, label='RF:AUC=%0.4f' % roc_auc)
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
max_features = arcpy.GetParameterAsText(3)
n_estimators = arcpy.GetParameterAsText(4)
min_samples_leaf = arcpy.GetParameterAsText(5)
min_samples_split = arcpy.GetParameterAsText(6)
max_depth = eval(max_depth)
max_features = eval(max_features)
n_estimators = eval(n_estimators)
min_samples_leaf = eval(min_samples_leaf)
min_samples_split = eval(min_samples_split)
row_patch_size = arcpy.GetParameterAsText(7)
col_patch_size = arcpy.GetParameterAsText(8)
channel = arcpy.GetParameterAsText(9)

AUC = []
test_acc, train_acc = [], []
with open(result_txt, "w") as f:
    pass
with open(result_txt, "a") as f:
    line = "max_depth\tmax_features\tn_estimators\tmin_samples_leaf\tmin_samples_split\tAUC\tTest_acc\tTrain_acc\tTrain_acc-Test_acc\n"
    f.write(line)

arcpy.AddMessage("******************************begain train!******************************")
arcpy.AddMessage("criterion=gini")
for i in max_depth:
    for j in max_features:
        for k in n_estimators:
            for l in min_samples_leaf:
                for m in min_samples_split:
                    model_save = model_save_path + '/md_' + str(i) + '_mf_' + str(j) + '_n_' + str(k) + '_msl_' + str(l) + '_mss_' + str(m)
                    if not os.path.exists(model_save):
                        os.makedirs(model_save)
                    train_result_txt = model_save + '/RF_train_result_txt.txt'
                    test_result_txt = model_save + '/RF_test_result_txt.txt'
                    roc_auc, testacc, trainacc = RF(data_root_path, model_save, test_result_txt, train_result_txt,row_patch_size, col_patch_size, channel, i, j,k,l,m)
                    with open(result_txt, "a") as f:
                        line = "%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (i, j, k,l,m,roc_auc, testacc, trainacc, trainacc - testacc)
                        f.write(line)
                    arcpy.AddMessage("********************md=" + str(i) + "," + "mf=" + str(j)+ "," + "n=" + str(k)+ "," + "msl=" + str(l)+ "," + "mss=" + str(m) + "RF train End!********************")

plt.rcParams['savefig.dpi'] = 350
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
result_df = pd.read_csv(result_txt, sep='\t', index_col=False)

fig, axm = plt.subplots(figsize=(16, 10))
# axm.set(aspect=1)# xlim=(-15, 15), ylim=(-20, 5)
axm.spines['top'].set_visible(False)
axm.spines['right'].set_visible(False)
axm.spines['left'].set_visible(False)
axm.spines['bottom'].set_visible(False)
axm.set_xticks([])
axm.set_yticks([])
# fig = plt.figure()
plt.rcParams['savefig.dpi'] = 350
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
# axins = zoomed_inset_axes(axm, zoom=0.5, loc='upper right')

xlabel=[' ']
ylabel=[' ']
max_depth = list(max_depth)
max_features = list(max_features)
xlabel=max_depth+xlabel
ylabel=max_features+ylabel

inset_ax = fig.add_axes([0.03, 0.06, 0.88, 0.90], facecolor='white')
plt.xticks([1.5,5.8,10,14.2,16],labels=xlabel,fontsize=15)
plt.yticks([1.5,4,6.5,9,10],labels=ylabel,fontsize=15)
plt.xlabel('max_depth')
plt.ylabel('max_features')
ax = []
for m in range(len(max_depth)*len(max_features)):
    ax.append(m)
k=1
for i in range(len(max_depth)):
    df = result_df[result_df['max_depth'] == max_depth[i]]
    for j in range(len(max_features)):
        df1 = df[df['max_features'] == max_features[j]]
        ax[k-1] = fig.add_subplot(len(max_depth),len(max_features),k,projection='3d')
        axc=ax[k-1].scatter3D(df1['n_estimators'], df1['min_samples_leaf'], df1['min_samples_split'], s=(df1['AUC']-min(df1['AUC']))/(max(df1['AUC'])-min(df1['AUC']))*100,c=df1['Train_acc-Test_acc'],cmap='rainbow',vmin=0, vmax=0.5)
        ax[k-1].set_xlabel('n_estimators')
        ax[k-1].set_ylabel('min_samples_leaf')
        ax[k-1].set_zlabel('min_samples_split')
        k=k+1
plt.subplots_adjust(left=0.01, top=0.95)
cax = plt.axes([0.93, 0.1, 0.025, 0.8])
cb=plt.colorbar(axc, cax=cax)
cb.ax.tick_params(labelsize=16)

plt.savefig(png_path, bbox_inches='tight')
plt.savefig(svg_path, bbox_inches='tight')
plt.show()
arcpy.AddMessage("*************************All parameter values training completed!*************************")



