library(ISLR)
library(dplyr)
source("functions/var_selection.R", echo = TRUE)
source("functions/mult_reg.R", echo = TRUE)

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



### 전체 변수에 대한 선택

hitters_dat = Hitters %>% na.omit() %>%
  dplyr::select(AtBat:NewLeague)

X_dummy = hitters_dat %>%
  dplyr::select(-Salary)
y_dummy = hitters_dat %>%
  dplyr::select(Salary)

X_dummy = dummy_var_gen(X_dummy)

best_subset_res = best_subset(X_dummy, y_dummy)

back_eli_res = back_eli(X_dummy, y_dummy, alpha_drop = 0.05, method = "adj_R2")

forward_sel_res = forward_sel(X_dummy, y_dummy, alpha_add = 0.05, method = "Mallow")

stepwise_sel_res = stepwise_sel(X_dummy, y_dummy, alpha_add = 0.05, alpha_drop = 0.10, method = "adj_R2")

library(parallel)
numCores <- detectCores()
numCores