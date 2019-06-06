dir = "E:\\OneDrive\\2018-2\\AI\\project\\2018_AI_PROJECT_201411644\\dataset"
setwd(dir)

library(caret)
library(irr)
library(C50)


df <-read.csv("clean_stdt_union.csv")
str(df)



str(df)
df$G3<-factor(df$G3)

set.seed(1234)
folds<-createFolds(df$G3, k=10)

cv_results <- lapply(folds, function(x){
  df_train <- df[-x,]
  df_test <- df[x, ]
  df_model <- C5.0(G3 ~. , data = df_train)
  df_pred <- predict(df_model, df_test)
  df_actual <- df_test$G3
  torf = df_actual == df_pred
  accuracy = sum(torf)/length(torf)
  kappa <- kappa2(data.frame(df_actual, df_pred))$value
  #return(kappa)
  return(accuracy)
})

mean(unlist(cv_results))





set.seed(1234)
m <- train(G3 ~ ., data = df, method = "C5.0")


m


ctrl <- trainControl(method = "cv", number = 10,
                     selectionFunction = "oneSE")

grid <- expand.grid(.model = "tree",
                    .trials = c(1, 5, 10, 15, 20),
                    .winnow = c("TRUE","FALSE"))


set.seed(1234)
m <- train(G3 ~ ., data = df, method = "C5.0",
           metric = "Kappa",
           trControl = ctrl,
           tuneGrid = grid)
m


library(ipred)
set.seed(1234)
ctrl <- trainControl(method = "cv", number = 10)
m<-train(G3 ~ ., data = df, method = "treebag",
         trControl = ctrl)
m


#1-8
library(adabag)
set.seed(1234)
adaboost_cv <- boosting.cv(G3 ~ ., data = df)

1-adaboost_cv$error

####clean된 데이터에 대해 set.seed(1234)로
#### adaboost :  47.41%



adaboost_cv$confusion
library(vcd)
Kappa(adaboost_cv$confusion)
### kappa also high


### stepwise
df$G3 <- as.numeric(df$G3)
full <- lm (G3 ~ .,data = df)
summary(full)
null <- lm (G3 ~ 1, data=df)
feature = step(data=df,null, direction = 'both', scope=list(upper=full))
f=toString(feature$call)
formula = substr(f, 5, nchar(f)-4)
formula
## important variable


set.seed(1234)
folds<-createFolds(df$G3, k=10)

## 변수 선택 후에 학습

cv_results <- lapply(folds, function(x){
  df_train <- df[-x,]
  df_test <- df[x, ]
  
  df_model <- lm(formula, data=df_train)
  df_pred <- round(predict(df_model, df_test))
  df_actual <- df_test$G3
  torf = df_actual == df_pred
  accuracy = sum(torf)/length(torf)
  kappa <- kappa2(data.frame(df_actual, df_pred))$value
  #return(kappa)
  return(accuracy)
})
mean(unlist(cv_results))


### 결론 adaboost가 가장 좋은 성능