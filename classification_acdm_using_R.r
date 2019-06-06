dir = "E:\\OneDrive\\2018-2\\AI\\project\\2018_AI_PROJECT_201411644\\dataset"
setwd(dir)

library(caret)
library(irr)
library(C50)

df <-read.csv("clean_stdt_acdm.csv")



str(df)
df$esp<-factor(df$esp)

set.seed(1234)
folds<-createFolds(df$esp, k=10)

cv_results <- lapply(folds, function(x){
  df_train <- df[-x,]
  df_test <- df[x, ]
  df_model <- C5.0(esp ~. , data = df_train)
  df_pred <- predict(df_model, df_test)
  df_actual <- df_test$esp
  torf = df_actual == df_pred
  accuracy = sum(torf)/length(torf)
  kappa <- kappa2(data.frame(df_actual, df_pred))$value
  #return(kappa)
  return(accuracy)
})

mean(unlist(cv_results))

#67.54



set.seed(1234)
m <- train(esp ~ ., data = df, method = "C5.0")
m

#59%

ctrl <- trainControl(method = "cv", number = 10,
                     selectionFunction = "oneSE")


grid <- expand.grid(.model = "tree",
                    .trials = c(1, 5, 10, 15, 20),
                    .winnow = c("TRUE","FALSE"))


set.seed(1234)
m <- train(esp ~ ., data = df, method = "C5.0",
           metric = "Kappa",
           trControl = ctrl,
           tuneGrid = grid)
m
#66.68%

library(ipred)
set.seed(1234)
ctrl <- trainControl(method = "cv", number = 10)
m<-train(esp ~ ., data = df, method = "treebag",
         trControl = ctrl)
m

#68.12  52.64


library(adabag)
set.seed(1234)

adaboost_cv <- boosting.cv(esp ~ ., data = df)
1-adaboost_cv$error

####clean된 데이터에 대해 set.seed(1234)로
#### adaboost 쓴게 63.36%



adaboost_cv$confusion
library(vcd)
Kappa(adaboost_cv$confusion)



### stepwise 
df$esp <- as.numeric(df$esp)
full <- lm (esp ~ .,data = df)
summary(full)
null <- lm (esp ~ 1, data=df)
feature = step(data=df,null, direction = 'both', scope=list(upper=full))
f = toString(feature$call)
formula = substr(f, 5, nchar(f)-4)




set.seed(1234)
folds<-createFolds(df$esp, k=10)

## 변수 선택 후에 학습

cv_results <- lapply(folds, function(x){
  df_train <- df[-x,]
  df_test <- df[x, ]

  df_model <- lm(formula, data=df_train)
  df_pred <- round(predict(df_model, df_test))
  df_actual <- df_test$esp
  torf = df_actual == df_pred
  accuracy = sum(torf)/length(torf)
  kappa <- kappa2(data.frame(df_actual, df_pred))$value
  #return(kappa)
  return(accuracy)
})
mean(unlist(cv_results))
## accuracy ==> 68.36

