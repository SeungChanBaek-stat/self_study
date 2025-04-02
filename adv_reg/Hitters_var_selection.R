library(ISLR)
library(dplyr)
source("functions/var_selection.R", echo = TRUE)

data(Hitters)

colnames(Hitters)

hit.dat <- Hitters %>% na.omit %>% select(Salary, AtBat, Hits, HmRun, Runs, RBI, Walks, Years, CAtBat,
                                          CHits, CHmRun, CRuns, CRBI, CWalks, PutOuts, Assists, Errors)

X = hit.dat[, -c(1)] ; y = hit.dat[, c(1)]

best_subset_res = best_subset(X, y)

back_eli_res = back_eli(X, y, alpha_drop = 0.15)

forward_sel_res = forward_sel(X, y, alpha_add = 0.15)

