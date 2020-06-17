**Task Description:**
Multiclass image classification

* Pre-extracted features from medical images
* Each image give as a continous feature vector of size 1000
* 4800 training and 4100 test feature vectors
* 3 classes labeled as {0,1,2}
* The same class distribution for training and test data


*Solution description:*
Dealing with unbalanced classes: The three classes were unbalanced with one class being three times the size of the two others. We tried many methods (SMOTE, upsampling, downsampling and bagging) but ended up with using the undersampling method. This is a method splitting the classes and then only using the equal number of each of the classes. The benefit of this is that we balance the data without introducing any form of bias and using the low variance of the smaller samples.

Method for classification: We tried out many different methods to make sure that the prediction would be both accurate and robust, without overfitting to the data. We ended up with using a support vector machine with a hard boarder line for the vectors between the classes. Even if there were other methods that in theory seems to be more complex this seemed to fit the best based on 5-fold cross-validation. Even not the bagging and voting (combining several different classifications and boosting methods) did improve the accuracy score.  

