library(ISLR)
library(dplyr)
source("functions/var_selection.R", echo = TRUE)

data(Hitters)

colnames(Hitters)

hit.dat <- Hitters %>% na.omit %>% select(Salary, AtBat, Hits, HmRun, Runs, RBI, Walks, Years, CAtBat,
                                          CHits, CHmRun, CRuns, CRBI, CWalks, PutOuts, Assists, Errors)

X = hit.dat[, -c(1)] ; y = hit.dat[, c(1)]

# print(class(X))
# print(class(y))
# p = dim(X)[2]
# 
# for (i in 1:p){
#   cat(i,"번째 열의 클래스 : ",  class(X[,i]))
# }
# class(X[,2])

res_hit = best_subset(X, y)

# y
# class(y)


