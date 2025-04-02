

source("functions/mult_reg.R", echo = FALSE)

#######################################################################################

total_subsets <- function(set) { 
  n <- length(set)
  masks <- 2^(1:n-1)
  res = sapply( 1:(2^n-1), function(u) set[ bitwAnd(u, masks) != 0 ] )
  
  return(res)
}

#######################################################################################

# 전역 선택함수
best_subset = function(X, y){
  library(glue)
  X = as.matrix(X)
  
  n = dim(X)[1] ; p = dim(X)[2]
  one = c(rep(1, n)) ; In = diag(1,n) ; Jn = one %*% t(one)
  SST = t(y) %*% (In - (1/n)*Jn) %*% y
  
  colname_vec = colnames(X)
  
  colname_vec_res = c(total_subsets(colname_vec))
  
  # 각 부분집합의 원소들을 오름차순 정렬
  colname_vec_res = lapply(colname_vec_res, sort, decreasing = FALSE)
  
  # 각 부분집합의 길이(원소 개수)에 따라 전체 리스트 정렬 (0개, 1개, 2개, …)
  colname_vec_res_sorted = colname_vec_res[order(sapply(colname_vec_res, length))]
  
  colvec = colname_vec_res_sorted
  
  # 총 부분집합수
  K = length(colvec)
  
  # 결과물 벡터
  SSE_vec = c(rep(0, K)) ; R2_a_vec = c(rep(0, K)) ; MSE_vec = c(rep(0, K))
  
  min_MSE = Inf ; min_MSE_index = NA
  max_R2_a = -1 ; max_R2_a_index = NA
  k = 0
  
  for (t in 1:K){
    subset_k = colvec[[t]]
    k_old = k
    k = length(subset_k)
    k_new = k
    if (k_new != k_old){
      print(glue("\n"))
    }
    X_k = X[, subset_k]
    X_k = cbind(one, X_k)
    X_ktX_k_inv = solve(t(X_k) %*% X_k)
    
    SSE_k = t(y) %*% (In - X_k %*% X_ktX_k_inv %*% t(X_k)) %*% y
    MSE_k = SSE_k / (n-k-1)
    R2_k = 1 - (SSE_k / SST)
    R2_ak = 1 - ((n-1)/(n-k-1)) * (1 - R2_k)
    
    SSE_vec[t] = SSE_k
    MSE_vec[t] = MSE_k
    R2_a_vec[t] = R2_ak
    
    if(MSE_vec[t] < min_MSE){
      min_MSE <- MSE_vec[t]
      min_MSE_index <- t
      print(glue("최소의 MSE_{k} = {min_MSE} 를 가지는 {t}번째 부분집합 : {paste(subset_k, collapse = ', ')}"))
    }
    
    if(R2_a_vec[t] > max_R2_a){
      max_R2_a <- R2_a_vec[t]
      max_R2_a_index <- t
      print(glue("최대의 R2_a{k} = {R2_ak} 를 가지는 {t}번째 부분집합 : {paste(subset_k, collapse = ', ')}"))
    }
    
    
    # print(glue("{t}번째 부분집합 : {paste(subset_k, collapse = ', ')}, SSE_{k} = {SSE_k}, R2_a{k} = {R2_ak} "))
    
  }
  
  return(list(col_vec = colvec, MSE_vec = MSE_vec, R2_a_vec = R2_a_vec,
              min_MSE_index = min_MSE_index, max_R2_a_index = max_R2_a_index))
}  
  


#######################################################################################
# 후진제거법

back_eli = function(X, y, alpha_drop = 0.15){
  library(glue)
  X = as.matrix(X)
  
  n = dim(X)[1] ; p = dim(X)[2]
  one = c(rep(1, n)) ; In = diag(1,n) ; Jn = one %*% t(one) ; 
  SST = t(y) %*% (In - (1/n)*Jn) %*% y
  
  
  end_operator = FALSE
  
  while(end_operator == FALSE){
    colname_vec = colnames(X)
    k = dim(X)[2] ; F_L_vec = c(rep(0, k)) ; X_F = cbind(one, X)
    
    SSE = t(y) %*% (In - X_F %*% solve(t(X_F) %*% X_F) %*% t(X_F)) %*% y
    MSE = SSE / (n - k - 1)    

    for(i in 1:k){
      index_sol = c(i) ; index_given = c(seq(from = 1, to = k, by = 1)) ; index_given = index_given[-index_sol]
      SS_beta = ASS_calc_char(X, y, index_vec = colname_vec,
                              index_sol = index_sol, index_given = index_given, coef = TRUE)
      F_L_element = SS_beta / MSE
      F_L_vec[i] = F_L_element
    }
    
    F_alpha_drop = qf(alpha_drop, 1, n - k - 1, lower.tail = FALSE)
    F_L = min(F_L_vec) ; drop_index = which.min(F_L_vec)
    
    if (F_L < F_alpha_drop){
      X = X[, -drop_index]
      print(glue("제거된 설명변수 : {colname_vec[drop_index]}, F_L = {F_L} < F_alpha_drop = {F_alpha_drop}"))
      k = k - 1
    }else{
      end_operator = TRUE
      print(glue("더 이상 제거할 설명변수 없음, 후진제거법 종료 : F_L = {F_L} > F_alpha_drop = {F_alpha_drop}"))
      # break
    }
    
  }
  final_var_vec = colnames(X)
  print(glue("최종 선택 변수 : {paste(final_var_vec, collapse = ', ')}"))
  
  return(list(X_out = X, final_var_vec = final_var_vec))
}


#######################################################################################
# 전진선택법

forward_sel = function(X, y, alpha_add = 0.15){
  library(glue)
  X = as.matrix(X)
  
  n = dim(X)[1] ; p = dim(X)[2]
  one = c(rep(1, n)) ; In = diag(1,n) ; Jn = one %*% t(one) ; 
  SST = t(y) %*% (In - (1/n)*Jn) %*% y
  
  
  end_operator = FALSE
  
  # initialize
  X_temp = cbind(one) ; k = 1
  colname_vec = colnames(X) 
  
  while(end_operator == FALSE){
    
    num_var = length(colname_vec)
    R2_vec = c(rep(0, num_var))
    MSE_vec = c(rep(0, num_var))
    for (i in 1:num_var){
      X_temp_null = X_temp
      X_temp_add = X[,colname_vec[i]]
      
      X_update = cbind(X_temp_null, X_temp_add)
      
      SSE = t(y) %*% (In - X_update %*% solve(t(X_update) %*% X_update) %*% t(X_update)) %*% y
      R2 = 1 - (SSE/SST) ; MSE = SSE / (n - k - 1)
      R2_vec[i] = R2
      MSE_vec[i] = MSE
    }
    
    max_R2_index = which.max(R2_vec)
    new_X = cbind(X[, colname_vec[max_R2_index]]) ; colnames(new_X) = colname_vec[max_R2_index]
    print(glue("결정계수(R2) = {R2_vec[max_R2_index]} 를 가장 크게 하는 설명변수 : {colname_vec[max_R2_index]}"))
    X_temp = cbind(X_temp, new_X) 
    temp_colname_vec = colnames(X_temp)
    temp_L = length(temp_colname_vec)
    
    # print(X_temp)
    # print(temp_colname_vec) ; print(temp_L)
    
    index_sol = c(temp_L) ; index_given = c(seq(from = 1, to = temp_L, by = 1)) ; index_given = index_given[-index_sol]
    SS_beta = ASS_calc_char(X_temp, y, index_vec = temp_colname_vec,
                            index_sol = index_sol, index_given = index_given, coef = FALSE)
    MSE = MSE_vec[max_R2_index]
    F_0 = SS_beta / MSE
    F_alpha_add = qf(alpha_add, 1, n - k - 1, lower.tail = FALSE)
    
    if (F_0 > F_alpha_add){
      print(glue("선택된 설명변수 : {colname_vec[max_R2_index]}, F_0 = {F_0} > F_alpha_add = {F_alpha_add}"))
      k = k + 1
      colname_vec = colname_vec[-max_R2_index]
      print(glue("\n"))
    }else{
      end_operator = TRUE
      del_index = dim(X_temp)[2]
      X_out = X_temp[,-c(1, del_index)]
      print(glue("더 이상 변수를 선택할 수 없음, 전진선택법 종료 : F_0 = {F_0} < F_alpha_add = {F_alpha_add}"))
      print(glue("\n"))
    }
    
  }
  final_var_vec = colnames(X_out)
  print(glue("최종 선택 변수 : {paste(final_var_vec, collapse = ', ')}"))
  return(list(X_out = X_out, final_var_vec = final_var_vec))
}