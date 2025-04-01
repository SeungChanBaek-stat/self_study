library(ISLR)
library(dplyr)
source("functions/var_selection.R", echo = TRUE)

data(Hitters)

colnames(Hitters)

hit.dat <- Hitters %>% na.omit %>% select(Salary, AtBat, Hits, HmRun, Runs, RBI, Walks, Years, CAtBat,
                                          CHits, CHmRun, CRuns, CRBI, CWalks, PutOuts, Assists, Errors)

X = hit.dat[, -c(1)] ; y = hit.dat[, c(1)]

res_hit = best_subset(X, y)


