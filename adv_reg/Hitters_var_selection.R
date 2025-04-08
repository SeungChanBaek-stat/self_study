library(ISLR)
library(dplyr)
source("functions/var_selection.R", echo = TRUE)

data(Hitters)

colnames(Hitters)

# hit.dat <- Hitters %>% na.omit %>% select(Salary, AtBat, Hits, HmRun, Runs, RBI, Walks, Years, CAtBat,
#                                           CHits, CHmRun, CRuns, CRBI, CWalks, PutOuts, Assists, Errors)

hit.dat <- Hitters %>% na.omit() %>% select(AtBat:CWalks, PutOuts:Salary)

last_col_index = dim(hit.dat)[2]

X = hit.dat[, -c(last_col_index)] ; y = hit.dat[, c(last_col_index)]

# best_subset_res = best_subset(X, y)

back_eli_res = back_eli(X, y, alpha_drop = 0.05, method = "adj_R2")

forward_sel_res = forward_sel(X, y, alpha_add = 0.05, method = "Mallow")

stepwise_sel_res = stepwise_sel(X, y, alpha_add = 0.05, alpha_drop = 0.10, method = "adj_R2")
