# Prerequisite
There are 2 training codes, "train.py" and the "train-1.py". The first one is designed to train the model with and without applying the transfer learning and the second one, we have added an additional step of sample selection to improve the accuracy.
<br>
The details of both the approaches are discussed in the document. 
<br>
Requirements.
<br>
Following packages are required to perform the training
1. pip install pandas 
2. pip install numpy
3. pip install tqdm
4. pip install matplotlib
5. pip install opencv-python
6. pip install -U scikit-learn
7. pip install tensorflow-gpu [for gpu] or pip install tensorflow [for cpu]

# Code execution
Before running the code, make sure that the dataset is placed in the "blazar_test/patches_candidates" and the lable file is placed under "blazar_test/" folder.
python Train.py

The same way we can execute the prediction code. Before running the code, create "models/" folder and place 20210704-163941 folder inside it.
Prediction code takes one random sample from the positive class and one from the negative class and then try to predict the target class.
python predict.py


