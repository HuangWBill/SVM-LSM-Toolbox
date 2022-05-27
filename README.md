# SVM-LSM-Toolbox
Landslide susceptibility mapping (LSM) is an important part of landslide risk assessment, but the process often needs to span multiple platforms, and the operation process is complex. We develops an efficient user-friendly toolbox including the whole process of LSM: **SVM-LSM toolbox**.

The toolbox realizes landslide susceptibility mapping based on a support vector machine (SVM), which can be integrated into **ArcGIS or ArcGIS Pro platform**. The toolbox includes three sub-toolboxes, namely:  
  *(1) influence factor production;*  
  *(2) dataset production and factor selection;*  
  *(3) model training and prediction.*

Influence factor production provides automatic calculation of DEM-related topographic factors, convert line vector data to continuous raster factor, and rainfall data processing. Factor selection uses PCC to calculate the correlation between factors, and IGR to calculate the contribution of factors to landslide occurrence. Dataset sample production includes automatic generation of non-landslides data, data sample production, and dataset split. The accuracy, precision, recall, F1 value, receiver operating characteristic (ROC) and area under curve (AUC) were used to evaluate the prediction ability of the model. In addition, we provide two methods, single process and multiprocessing, to generate LSM. The prediction efficiency of multiprocessing is much higher than that of single process.

## It includes the *toolbox* folder and the *manual* folder. The details are as follows:

The *toolbox* folder contains a **.tbx** format toolbox and  two **dist** folders that necessary for the multiprocessing prediction tools to run.

The *manual* folder contains two **PDF** files.

  One is **"installation instructions and toolbox introduction"**, which introduces the installation of the toolbox and the instructions for the use of each tool。

  The other is the **"usage process (case)"**, which introduces the whole process of use in a specific case (Wuqi County, Yan'an City, Shaanxi Province, China) for reference.
  
## Others:

### ***Relevant literature: DOI:XXXXXX***

### ***If you use this toolkit in your research, please cite to the literature as follows:***  
***XXXXXXXXXX***

### ***If you have any problems with your usage, you can send an email to huangwubiao@chd.edu.cn.***
