# Double-Bottom-Classifier-
FOR A DEMONSTRATION:
  Download the TechnicalAnalysisCNN.zip folder and unzip. 

This repository contains peripherial code and a convolutional neural network for identifing double bottoms in stock pricing. Run DEMO.py inside the files folder. All libraries listed at the top of the file must be installed. This project is based on a peak classifier CNN created by Ian Van Horn and Mason Brady for classifing optical-eletrical signals. Data for this project was annotated by Justin Kane.

Code files:


  Model2:
    Containes the arcatecture for the detection CNN. Also containes a basic evaluation code for precision/recall and confusion matrix. 
    
  Annotate2:
    Containes perepherial code, this includes annotation script, visualization script, data cleaning script, data merging scrpit, and data augmentation script.

MERGE folder: Contains set of stock prices downloaded from Yahoo Finance. These files are merged and annotated to create a dataset.

SPY_TRU: The truths file with augmntation used for training

MODELS: This folder containes a model file, its validation set, and the corrosponding confusion matrix



