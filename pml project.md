PRACTICAL MACHINE LEARNING PROJECT
========================================================


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```


```r
readTrain<-read.csv('pml-training.csv')
dim(readTrain)
```

```
## [1] 19622   160
```

```r
readTest<-read.csv('pml-testing.csv')
dim(readTest)
```

```
## [1]  20 160
```

The loaded data seems very untidy,therefore cleaning the data will be the first step in our analysis.


```r
train_rem=colSums(is.na(readTrain))==0
test_rem=colSums(is.na(readTest))==0

tidy_train<-readTrain[,train_rem]
dim(tidy_train)
```

```
## [1] 19622    93
```

```r
tidy_test<-readTest[,test_rem]
dim(tidy_test)
```

```
## [1] 20 60
```

The first seven variables are not relevant to our analysis and have little or no impact on our outcome


```r
real_train<-tidy_train[,-c(1:7)]
dim(real_train)
```

```
## [1] 19622    86
```

```r
real_test<-tidy_test[,-c(1:7)]
dim(real_test)
```

```
## [1] 20 53
```

Now, split the real_train dataframe into training and testing set


```r
set.seed(999)
inTrain<-createDataPartition(real_train$classe,p=0.75,list=F)
training<-real_train[inTrain,]
testing<-real_train[-inTrain,]
dim(training)
```

```
## [1] 14718    86
```

```r
dim(testing)
```

```
## [1] 4904   86
```
first Model: Decision Tree


```r
Tree_Model<-train(classe~.,data=training,method='rpart')
```

```
## Warning: model fit failed for Resample02: cp=0.03389 Error : cannot allocate vector of size 115 Kb
```

```
## Warning: model fit failed for Resample03: cp=0.03389 Error : cannot allocate vector of size 384.9 Mb
```

```
## Warning: model fit failed for Resample04: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample05: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample06: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample07: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample08: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample09: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample10: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample11: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample12: cp=0.03389 Error : cannot allocate vector of size 770.0 Mb
```

```
## Warning: model fit failed for Resample13: cp=0.03389 Error : cannot allocate vector of size 384.9 Mb
```

```
## Warning: model fit failed for Resample14: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample15: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample16: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample17: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample18: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample19: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample20: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample21: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample22: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample23: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample24: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample25: cp=0.03389 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info =
## trainInfo, : There were missing values in resampled performance measures.
```

```
## Error: cannot allocate vector of size 769.9 Mb
```

```
## Timing stopped at: 5.11 0.33 6.09
```

```r
fancyRpartPlot(Tree_Model$finalModel)
```

```
## Error in fancyRpartPlot(Tree_Model$finalModel): object 'Tree_Model' not found
```

Lets predict classe in the test data using the decision tree model


```r
tree_predict<-predict(Tree_Model,testing)
```

```
## Error in predict(Tree_Model, testing): object 'Tree_Model' not found
```

```r
confusionMatrix(tree_predict,testing$classe)
```

```
## Error in confusionMatrix(tree_predict, testing$classe): object 'tree_predict' not found
```

**RANDOM FOREST MODEL**


```r
forest_model<-train(classe~.,data=training,method='rf',verbose=F)
```

```
## Warning: model fit failed for Resample01: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample01: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample01: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample02: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample02: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample02: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample03: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample03: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample03: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample04: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample04: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample04: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample05: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample05: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample05: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample06: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample06: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample06: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample07: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample07: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample07: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample08: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample08: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample08: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample09: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample09: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample09: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample10: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample10: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample10: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample11: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample11: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample11: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample12: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample12: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample12: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample13: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample13: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample13: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample14: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample14: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample14: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample15: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample15: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample15: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample16: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample16: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample16: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample17: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample17: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample17: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample18: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample18: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample18: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample19: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample19: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample19: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample20: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample20: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample20: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample21: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample21: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample21: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample22: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample22: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample22: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample23: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample23: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample23: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample24: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample24: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample24: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample25: mtry=   2 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample25: mtry= 117 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning: model fit failed for Resample25: mtry=6856 Error : cannot allocate vector of size 769.9 Mb
```

```
## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info =
## trainInfo, : There were missing values in resampled performance measures.
```

```
## Something is wrong; all the Accuracy metric values are missing:
##     Accuracy       Kappa    
##  Min.   : NA   Min.   : NA  
##  1st Qu.: NA   1st Qu.: NA  
##  Median : NA   Median : NA  
##  Mean   :NaN   Mean   :NaN  
##  3rd Qu.: NA   3rd Qu.: NA  
##  Max.   : NA   Max.   : NA  
##  NA's   :3     NA's   :3
```

```
## Error: Stopping
```

```r
print(forest_model)
```

```
## Error in print(forest_model): object 'forest_model' not found
```

```r
plot(forest_model,main='Accuracy by number of predictors used')
```

```
## Error in plot(forest_model, main = "Accuracy by number of predictors used"): object 'forest_model' not found
```
