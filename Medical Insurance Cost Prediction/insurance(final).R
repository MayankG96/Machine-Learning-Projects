

# Getting the required libraries
library(tidyverse)
library(randomForest)
library(caret)
library(gbm)
library(caTools)
library(psych)
library(tree)
library(class)
library(kknn)
library(ggplot2)
library(gplots)

# Reading the dataset
# Input the dataset here 
Insurance <- read.csv("C:/Users/Mayank/OneDrive/Desktop/ML project/insurance.csv")



#EDA
#Visualizing data distributions of features

#Age
hist(Insurance$age, 
     main = "Distribution of Age",
     xlab = "Age",
     ylab = "Count",
     col = "blue",
     border = "white")

#sex
sex_dist = table(Insurance$sex)
pie(sex_dist, 
    main = "Sex Distribution",
    col = c("red", "lightblue") ,
    labels = paste0(names(sex_dist), " (", sex_dist, ")"),
    cex = 0.6,
    radius = 0.6)

#BMI
hist(Insurance$bmi, 
     main = "Distribution of BMI",
     xlab = "BMI",
     ylab = "Count",
     col = "blue",
     border = "white")

#Children
hist(Insurance$children, 
     main = "Distribution of Number of Children",
     xlab = "Number of Children",
     ylab = "Count",
     col = "blue")

#Smoker
smoker_dist = table(Insurance$smoker)
pie(smoker_dist, 
    main = "Smoker Distribution",
    col = c("green", "red") ,
    labels = paste0(names(smoker_dist), " (", smoker_dist, ")"),
    cex = 0.6,
    radius = 0.6)

#Charges
hist(Insurance$charges, 
     main = "Distribution of Charges",
     xlab = "Charges",
     ylab = "Count",
     col = "blue",
     border = "white")

# Charges by smoker
boxplot(charges ~ smoker, data = Insurance,
        main = "Charges by Smoker status",
        xlab = "Smoker",
        ylab = "Charges",
        col = c("red", "green"))

# Relationship between age and charges
plot(Insurance$age, Insurance$charges,
     main = "Relation between charges and age",
     xlab = "Age",
     ylab = "Charges",
     col = "brown")

# Lets see relationships beetween BMI and charges
plot(Insurance$bmi, Insurance$charges,
     main = "Relation between charges and BMI",
     xlab = "BMI",
     ylab = "Charges",
     col = "#006172",
     pch = 16)

# Check distribution of charges by sex
boxplot(charges ~ sex, data = Insurance,
        main = "Relation between Charges and Sex",
        xlab = "Sex",
        ylab = "Charges",
        col = c("red", "green"))

# Check distribution of charges by number of children
boxplot(charges ~ children, data = Insurance,
        main = "Relation between charges and Number of children",
        xlab = "Number of children",
        ylab = "Charges",
        col = c("maroon", "green", "red", "pink", "blue", 'magenta'))

# Distribution of charges by region
boxplot(charges ~ region, data = Insurance,
        main = "Relation between charges and Region",
        xlab = "Region",
        ylab = "Charges",
        col = c("darkblue", "yellow", "red", "green"))


#correlation Matrix
# Create numeric dataframe
df_cor <- Insurance
df_cor$sex <- ifelse(Insurance$sex == "male",1,0)
df_cor$smoker <- ifelse(Insurance$smoker == "yes", 1, 0)
df_cor$region <- as.numeric(factor(Insurance$region, levels = unique(Insurance$region)))


# Calculate the correlation matrix
correlation_matrix <- cor(df_cor)

heatmap.2(correlation_matrix, 
          main = "Correlation Heatmap",
          col = colorRampPalette(c("white", "darkblue"))(100),
          key = FALSE,    
          symkey = FALSE,
          trace="none",
          cexCol = 1.3,    
          cexRow = 1.3,    
          srtCol = 45,      
          cellnote = round(correlation_matrix, 2),   
          notecol = "white",    
          notecex = 1)


ggplot(data = Insurance) +aes(smoker,charges)+ geom_boxplot() + ggtitle("Boxplot of Medical Charges by Smoking Status")

# smokers spends a lot more in terms of medical expenses compared to non-smokers by almost 4x
hist(Insurance$charges, main='Histogram of Medical Costs', labels = TRUE, xlab = 'Individual medical costs billed by health insurance', ylab='No of people')

#checking correlations

#feature transformation
Insurance$sex <- ifelse(Insurance$sex == "male",1,0)
Insurance$smoker <- ifelse(Insurance$smoker == "yes", 1, 0)
Insurance$region <- as.numeric(factor(Insurance$region, levels = unique(Insurance$region)))

#### Linear Regression ####

# Split the data into training and test sets
set.seed(100)
split = createDataPartition(y=Insurance$charges,p=0.8,list = FALSE )
x_train <- Insurance[split, -7]
y_train <- Insurance$charges[split]
x_test <- Insurance[-split, -7]
y_test <- Insurance$charges[-split]

# Fit the Linear Regression model 
Lin_reg <- lm(charges ~ ., data = data.frame(charges = y_train, x_train))
print(coef(Lin_reg)[1])
print(coef(Lin_reg)[-1])

# Predict on the test set using Linear Regression and Calculate RMSE
y_pred <- predict(Lin_reg, newdata = data.frame(x_test))
print(cor(y_pred, y_test)^2)
rmse_linear <- sqrt(mean((y_pred - y_test)^2))
print(rmse_linear)
plot(y_pred,y_test)

#Adding some extra features 
Insurance2 <- Insurance
Insurance2$age2 <- Insurance2$age^2
Insurance2$bmi30 <- ifelse(Insurance2$bmi>=30, 1, 0)

# Split the data into training and test sets
split = createDataPartition(y=Insurance2$charges,p=0.8,list = FALSE )
x_train <- Insurance2[split, (!names(Insurance2) %in% c('charges'))]
y_train <- Insurance2$charges[split]
x_test <- Insurance2[-split, (!names(Insurance2) %in% c('charges'))]
y_test <- Insurance2$charges[-split]

# Fit the Linear Regression model 
Lin_reg <- lm(charges~.+ bmi30*smoker, data=data.frame(charges= y_train, x_train))
print(coef(Lin_reg)[1])
print(coef(Lin_reg)[-1])

# Predict on the test set using Linear Regression and Calculate RMSE
y_pred <- predict(Lin_reg, newdata = data.frame(x_test))
print(cor(y_pred, y_test)^2)
rmse_linear <- sqrt(mean((y_pred - y_test)^2))
print(rmse_linear)
plot(y_pred,y_test)


#### Boosting #####

# Split the data into training and test sets
set.seed(100)
sample.data <- sample.split(Insurance$charges, SplitRatio = 0.70)
train.set <- subset(Insurance, sample.data == TRUE)
test.set <- subset(Insurance, sample.data == FALSE)

# Define the training control for cross-validation
ctrl <- trainControl(method = "cv", number = 7)

# Define the hyperparameter grid for Gradient Boosting
gbm_grid <- expand.grid(
  n.trees = c(100, 200, 300),
  interaction.depth = c(3, 5, 7),
  shrinkage = c(0.01, 0.1, 0.2),
  n.minobsinnode = c(10, 20, 30)
)

# Fit the Gradient Boosting model with hyperparameter tuning
gbm_model <- train(charges ~ ., data = train.set,
                   method = "gbm",
                   trControl = ctrl,
                   tuneGrid = gbm_grid)

# Predict on the test set using Gradient Boosting
gbm_predictions <- predict(gbm_model, newdata = test.set)

# Calculate RMSE for Gradient Boosting
rmse_gbm <- sqrt(mean((test.set$charges - gbm_predictions)^2))
rmse_gbm
best_params <- gbm_model$bestTune
print(best_params)
gbm_feature_importance <- summary(gbm_model)
print(gbm_feature_importance)
barplot(height = gbm_feature_importance$rel.inf, names.arg = gbm_feature_importance$var,
        main = "Gradient Boosting Feature Importance for All Predictors",
        ylab = "Relative Influence", las = 2,
        cex.names = 0.8, col = "blue")

#### Boosting 2 ####

#Adding some extra features 
Insurance2 <- Insurance
Insurance2$age2 <- Insurance2$age^2
Insurance2$bmi30 <- ifelse(Insurance2$bmi>=30, 1, 0)

# Split the data into training and test sets
split = split(Insurance2, sample(rep(1:3,times=c(803,267,268))))
train.set <- split$'1'
val.set <- split$'2'
test.set <- split$'3'

# Define the hyperparameter grid for Gradient Boosting
idv = c(5)
ntv = c(5000,1000)
lamv=c(.00165,.00135,.00125)
parmb = expand.grid(idv,ntv,lamv)
colnames(parmb) = c('tdepth','ntree','lam')
print(parmb)
nset = nrow(parmb)
olb = rep(0,nset)
ilb = rep(0,nset)
bfitv = vector('list',nset)

# Fit the Gradient Boosting model with hyperparameter tuning
for(i in 1:nset) {
  cat('doing boost ',i,' out of ',nset,'\n')
  tempboost = gbm(charges~.,data=train.set,distribution='gaussian',
                  interaction.depth=parmb[i,1],n.trees=parmb[i,2],shrinkage=parmb[i,3])
  ifit = predict(tempboost,n.trees=parmb[i,2])
  ofit=predict(tempboost,newdata=test.set,n.trees=parmb[i,2])
  olb[i] = sum((val.set$charges-ofit)^2)
  ilb[i] = sum((train.set$charges-ifit)^2)
  bfitv[[i]]=tempboost
}
ilb = round(sqrt(ilb/nrow(train.set)),3); olb = round(sqrt(olb/nrow(val.set)),3)

#Print losses
print(cbind(parmb,olb,ilb))

#Write val preds
iib=which.min(olb)
theb = bfitv[[iib]] 
thebpred = predict(theb,newdata=val.set,n.trees=parmb[iib,2])

#Fit on train+val
boost_trainval = rbind(train.set,val.set)
ntrees=5000
finb = gbm(charges~.,data=boost_trainval,distribution='gaussian',
           interaction.depth=4,n.trees=ntrees,shrinkage=.2)
finbpred=predict(finb,newdata=test.set,n.trees=ntrees)

#Plot y vs yhat for test data and compute rmse on test.
finbrmse = sqrt(sum((test.set$charges-finbpred)^2)/nrow(test.set))
cat('finbrmse: ',finbrmse,'\n')
plot(test.set$charges,finbpred,xlab='test charges',ylab='boost pred')
abline(0,1,col='red',lwd=2)

#Plot variable importance
p=ncol(train.set)-1 #want number of variables for later
vsum=summary(finb) #this will have the variable importance info
row.names(vsum)=NULL #drop varable names from rows.

#Write variable importance table
cat('\\begin{verbatim}\n')
print(vsum)
cat('\\end{verbatim}\n')

#Plot variable importance
plot(vsum$rel.inf,axes=F,pch=16,col='red')
axis(1,labels=vsum$var,at=1:p)
axis(2)
for(i in 1:p) lines(c(i,i),c(0,vsum$rel.inf[i]),lwd=4,col='blue')

#### KNN ####

#Adding some extra features 
Insurance2 <- Insurance
Insurance2$age2 <- Insurance2$age^2
Insurance2$age <- log(Insurance2$age)
Insurance2$bmi <- log(Insurance2$bmi)
Insurance2$charges <- log(Insurance2$charges)

# Split the data into training and test sets
set.seed(100)
n = nrow(Insurance2)
n1 = round(0.8*n)
ii = sample(1:n,n)
train.set = Insurance2[ii[1:n1],]
test.set = Insurance2[ii[(n1+1):n],]

x_train = train.set[,c(1:6,8)]
y_train = train.set[,7]
x_test = test.set[,c(1:6,8)]
y_test = test.set[,7]

# Define the training control for cross-validation
trControl <- trainControl(method  = "cv",
                          number  = 10)
# Fit the KNN  model 
fit <- train(charges ~ .+smoker*age+bmi*age,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:15),
             trControl  = trControl,
             metric     = "RMSE",
             data       = train.set)

fit

# Predict on the test set using Linear Regression and Calculate RMSE
ypred_knn <- exp(predict(fit, test.set))
rmse_knn = sqrt(mean((exp(y_test)-ypred_knn)^2))
rmse_knn

#### Random Forest ####

#Adding some extra features 
Insurance2 <- Insurance
Insurance2$age_sq <- (Insurance2$age)^2
Insurance2$age <- log(Insurance2$age)
Insurance2$bmi <- log(Insurance2$bmi)
Insurance2$charges <- log(Insurance2$charges)

# Split the data into training and test sets
set.seed(100)
n = nrow(Insurance2)
n1 = round(0.8*n)
ii = sample(1:n,n)
train.set = Insurance2[ii[1:n1],]
test.set = Insurance2[ii[(n1+1):n],]
x_train = train.set[,c(1:6,8)]
y_train = train.set[,7]
x_test = test.set[,c(1:6,8)]
y_test = test.set[,7]

# Define the parameters for Random Forest
set.seed(100)
p=ncol(train.set)-1
mtryv = c(p,round(sqrt(p)),5)
ntreev=seq(90, 110, by = 5)

# Define the training control for cross-validation
trControl <- trainControl(method  = "cv",
                          number  = 10)

parmrf = expand.grid(ntreev)
nset = nrow(parmrf)
rmse_rf = rep(0,nset)
mtry_rf = rep(0,nset)
ntreev_rf=rep(0,nset)

# Fit the Random Forest model 
for(i in 1:nset) {
  # ntreev=parmrf[i,1]
  # ntreev=100
  fit <- train(charges ~ .+smoker*age+bmi*age,
               method     = "rf",
               tuneGrid   = expand.grid(mtry=mtryv),
               trControl  = trControl,
               metric     = "RMSE",
               # ntreev     = ntreev,
               data       = train.set)
  
  fit
  best_mtry <- fit$bestTune$mtry
  ypred_rf <- exp(predict(fit, test.set))
  rmse_rf[i] = sqrt(mean((exp(y_test)-ypred_rf)^2))
  mtry_rf[i] = best_mtry
  # ntreev_rf[i] = ntreev
}

#Getting the RMSE
# print(cbind(mtry_rf,ntreev_rf,rmse_rf))
print(cbind(mtry_rf,rmse_rf))
