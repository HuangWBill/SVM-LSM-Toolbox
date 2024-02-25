# SVM-LSM-Toolbox
Landslide susceptibility mapping (LSM) is an important part of landslide risk assessment, but the process often needs to span multiple platforms, and the operation process is complex. We develops an efficient user-friendly toolbox including the whole process of LSM: **SVM-LSM toolbox**.

The toolbox realizes landslide susceptibility mapping based on a support vector machine (SVM), which can be integrated into **ArcGIS 10.1 (or higher) or ArcGIS Pro platform**. The toolbox includes three sub-toolboxes, namely:  
  *(1) influence factor production;*  
  *(2) dataset production and factor selection;*  
  *(3) model training and prediction.*

Influence factor production provides automatic calculation of DEM-related topographic factors, convert line vector data to continuous raster factor, and rainfall data processing. Factor selection uses PCC to calculate the correlation between factors, and IGR to calculate the contribution of factors to landslide occurrence. Dataset sample production includes automatic generation of non-landslides data, data sample production, and dataset split. The accuracy, precision, recall, F1 value, receiver operating characteristic (ROC) and area under curve (AUC) were used to evaluate the prediction ability of the model. In addition, we provide two methods, single process and multiprocessing, to generate LSM. The prediction efficiency of multiprocessing is much higher than that of single process.

## It includes the *Article.pdf* file, the *toolbox* folder and the *manual* folder. The details are as follows:

the *Article.pdf* file is an article related to SVM-LSM toolbox, entitled "An Efficient User-Friendly Integration Tool for Landslide Susceptibility Mapping Based on Support Vector Machines: SVM-LSM Toolbox".

The *toolbox* folder contains a **.tbx** format toolbox and  two **dist** folders that necessary for the multiprocessing prediction tools to run.

The *manual* folder contains two **PDF** files.

  One is **"installation instructions and toolbox introduction"**, which introduces the installation of the toolbox and the instructions for the use of each tool。

  The other is the **"usage process (case)"**, which introduces the whole process of use in a specific case (Wuqi County, Yan'an City, Shaanxi Province, China) for reference.
  
## Others:

### ***Relevant literature: DOI: https://doi.org/10.3390/rs14143408***

### ***If you use this toolbox in your research, please cite to the literature as follows:***  
***Huang, W.; Ding, M.; Li, Z.; Zhuang, J.; Yang, J.; Li, X.; Meng, L.; Zhang, H.; Dong, Y. An Efficient User-Friendly Integration Tool for Landslide Susceptibility Mapping Based on Support Vector Machines: SVM-LSM Toolbox. Remote Sens. 2022, 14, 3408. https://doi.org/10.3390/rs14143408***

### ***If you have any problems with your usage, you can send an email to huangwubiao@chd.edu.cn.***

## Update records：
### February 27, 2023：
  According to the feedback of ***"ValueError: invalid literal for int() with base 10:'0.' "*** error in the *"Model Training and Performance Evaluation of SVM"*, *"Landslide Susceptibility Map Prediction (Single process)"* and *"Landslide Susceptibility Map Prediction (Multiprocessing)"* tool during the user's use, the toolbox was updated on **February 27, 2023** to solve the above problems.
### August 22, 2023：
  According to the feedback of users, Add ***Supplementary instructions (required reading).txt*** in ***manual*** folder. This *.txt* file can help users better use the toolbox. The file content is as follows:

  **Note: During the use of the toolbox, all files must be loaded from the path, cannot be directly selected from the software interface’s layers, otherwise an error will occur.**

## Notification:
Hello, teachers and scholars!

Due to the change of my research direction and other reasons, I may not maintain the SVM-LSM Toolbox in the near future. If you have some problems that cannot be solved, you can join the WeChat group below (the QR code is valid for 7 days) to communicate and learn from each other. We deeply apologize for not being able to reply to your emails in time!
Wishing you all good luck in your research and good health!

SVM-LSM Toolbox Developer

10th August 2023

![WeChat](https://github.com/HuangWBill/SVM-LSM-Toolbox/assets/76198298/e9477c87-ae2a-48f5-9349-18e5fcddf0ef)
