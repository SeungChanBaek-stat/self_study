

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
  library(parallel)
  X = as.matrix(X) ; y = as.matrix(y)
  
  n = dim(X)[1] ; p = dim(X)[2]
  one = c(rep(1, n)) ; In = diag(1,n) ; Jn = one %*% t(one)
  SST = t(y) %*% (In - (1/n)*Jn) %*% y
  SSE = t(y) %*% (In - X %*% solve(t(X) %*% X) %*% t(X)) %*% y
  MSE = SSE / (n - p - 1)
  
  colname_vec = colnames(X)
  
  colname_vec_res = c(total_subsets(colname_vec))
  
  # 각 부분집합의 원소들을 오름차순 정렬
  colname_vec_res = lapply(colname_vec_res, sort, decreasing = FALSE)
  
  # 각 부분집합의 길이(원소 개수)에 따라 전체 리스트 정렬 (0개, 1개, 2개, …)
  colname_vec_res_sorted = colname_vec_res[order(sapply(colname_vec_res, length))]
  
  colvec = colname_vec_res_sorted
  
  # 총 부분집합수
  K = length(colvec)
  
  # 중간계산 벡터
  SSE_vec = c(rep(0, K)) ; R2_a_vec = c(rep(0, K)) ; R2_vec = c(rep(0, K)) ; MSE_vec = c(rep(0, K)) ; C_k_vec = c(rep(0, K))
  
  # 결과물 벡터
  MSE_k_res = c(rep(0, p)) ; R2_k_res = c(rep(0, p))
  adj_R2_k_res = c(rep(0, p)) ; C_k_res = c(rep(0, p))
  
  
  
  min_MSE = Inf ; min_MSE_index = NA
  max_R2 = -1 ; max_R2_index = NA
  max_R2_a = -1 ; max_R2_a_index = NA
  min_C_k = Inf ; min_C_k_index = NA
  k = 1
  
  for (t in 1:K){
    subset_k = colvec[[t]]
    
    var_nums = length(subset_k)
    
    if (k < var_nums | k == p){
      MSE_k_res[k] = min_MSE
      R2_k_res[k] = max_R2
      adj_R2_k_res[k] = max_R2_a
      C_k_res[k] = min_C_k
      print(glue("k = {k} 일때 MSE기준 변수선택 : {paste(colvec[[min_MSE_index]], collapse = ', ')}, MSE = {MSE_k_res[k]}"))
      print(glue("k = {k} 일때 R2기준 변수선택 : {paste(colvec[[max_R2_index]],  collapse = ', ')}, R2 = {R2_k_res[k]}"))
      print(glue("k = {k} 일때 adj_R2기준 변수선택 : {paste(colvec[[max_R2_a_index]],  collapse = ', ')}, adj_R2 = {adj_R2_k_res[k]}"))
      print(glue("k = {k} 일때 C_k기준 변수선택 : {paste(colvec[[min_C_k_index]],  collapse = ', ')}, C_k = {C_k_res[k]}"))
      
      print(glue("\n"))
      k = k + 1
      print(glue("현재 k = {k}"))
    }

    X_k = X[, subset_k]
    X_k = cbind(one, X_k)
    X_ktX_k_inv = solve(t(X_k) %*% X_k)
    
    SSE_k = t(y) %*% (In - X_k %*% X_ktX_k_inv %*% t(X_k)) %*% y
    MSE_k = SSE_k / (n-k-1)
    R2_k = 1 - (SSE_k / SST)
    R2_ak = 1 - ((n-1)/(n-k-1)) * (1 - R2_k)
    C_k = (SSE_k / MSE) + 2*(k+1) - n
    
    SSE_vec[t] = SSE_k
    MSE_vec[t] = MSE_k
    R2_vec[t] = R2_k
    R2_a_vec[t] = R2_ak
    C_k_vec[t] = C_k
    
    if(MSE_vec[t] < min_MSE){
      min_MSE <- MSE_vec[t]
      min_MSE_index <- t
      # print(glue("최소의 MSE_{k} = {min_MSE} 를 가지는 {t}번째 부분집합 : {paste(subset_k, collapse = ', ')}"))
    }
    
    if(R2_vec[t] > max_R2){
      max_R2 <- R2_vec[t]
      max_R2_index <- t
      # print(glue("최대의 R2_{k} = {R2_k} 를 가지는 {t}번째 부분집합 : {paste(subset_k, collapse = ', ')}"))      
    }
    
    if(R2_a_vec[t] > max_R2_a){
      max_R2_a <- R2_a_vec[t]
      max_R2_a_index <- t
      # print(glue("최대의 R2_a{k} = {R2_ak} 를 가지는 {t}번째 부분집합 : {paste(subset_k, collapse = ', ')}"))
    }
    
    if(C_k_vec[t] < min_C_k){
      min_C_k =  C_k_vec[t]
      min_C_k_index = t
      
    }
    
    
    # print(glue("{t}번째 부분집합 : {paste(subset_k, collapse = ', ')}, SSE_{k} = {SSE_k}, R2_a{k} = {R2_ak} "))
    
  }
  MSE_k_res[k] = min_MSE
  R2_k_res[k] = max_R2
  adj_R2_k_res[k] = max_R2_a
  C_k_res[k] = min_C_k
  print(glue("k = {k} 일때 MSE기준 변수선택 : {paste(colvec[[min_MSE_index]], collapse = ', ')}, MSE = {MSE_k_res[k]}"))
  print(glue("k = {k} 일때 R2기준 변수선택 : {paste(colvec[[max_R2_index]],  collapse = ', ')}, R2 = {R2_k_res[k]}"))
  print(glue("k = {k} 일때 adj_R2기준 변수선택 : {paste(colvec[[max_R2_a_index]],  collapse = ', ')}, adj_R2 = {adj_R2_k_res[k]}"))
  print(glue("k = {k} 일때 C_k기준 변수선택 : {paste(colvec[[min_C_k_index]],  collapse = ', ')}, C_k = {C_k_res[k]}"))
  
  k = c(seq(from = 1, to = p, by = 1))
  table_out = data.frame(k, MSE_k_res, R2_k_res, adj_R2_k_res, C_k_res)
  colnames(table_out) = c("k", "MSE", "R2", "adj_R2", "C_k")
  
  plot(C_k_vec, type="b", xlab="Subset index", ylab="C_k")
  
  return(table_out)
}  
  


#######################################################################################
# 후진제거법

back_eli = function(X, y, alpha_drop = 0.15, method = c("MSE", "R2", "adj_R2", "Mallow")){
  library(glue)
  X = as.matrix(X) ; y = as.matrix(y)
  
  n = dim(X)[1] ; p = dim(X)[2]
  one = c(rep(1, n)) ; In = diag(1,n) ; Jn = one %*% t(one) ; 
  SST = t(y) %*% (In - (1/n)*Jn) %*% y
  
  
  end_operator = FALSE
  
  while(end_operator == FALSE){
    colname_vec = colnames(X)
    k = dim(X)[2] ; X_F = cbind(one, X)
    
    SSE = t(y) %*% (In - X_F %*% solve(t(X_F) %*% X_F) %*% t(X_F)) %*% y
    MSE = SSE / (n - k - 1)
    
    if (method == "MSE"){
      F_L_vec = c(rep(0, k))
      for(i in 1:k){
        index_sol = c(i) ; index_given = c(seq(from = 1, to = k, by = 1)) ; index_given = index_given[-index_sol]
        SS_beta = ASS_calc_char(X, y, index_vec = colname_vec,
                                index_sol = index_sol, index_given = index_given, coef = TRUE)
        F_L_element = SS_beta / MSE
        F_L_vec[i] = F_L_element
      }
      
      drop_index = which.min(F_L_vec)
      
    }else if (method == "R2"){
      R2_vec = c(rep(0, k))
      
      for(i in 1:k){
        X_i = X_F[, -c(i+1)]
        SSE_i = t(y) %*% (In - X_i %*% solve(t(X_i) %*% X_i) %*% t(X_i)) %*% y
        R2_i = 1 - (SSE_i / SST)
        R2_vec[i] = R2_i
      }
      drop_index = which.max(R2_vec)
      
    }else if(method == "adj_R2"){
      R2_a_vec = c(rep(0, k))
      
      for(i in 1:k){
        X_i = X_F[, -c(i+1)]
        SSE_i = t(y) %*% (In - X_i %*% solve(t(X_i) %*% X_i) %*% t(X_i)) %*% y
        R2_i = 1 - (SSE_i / SST)
        R2_ai = 1 - ((n - 1)/(n-k-1)) * (1 - R2_i)
        R2_a_vec[i] = R2_ai
      }
      drop_index = which.max(R2_a_vec)
      
    }else if(method == "Mallow"){
      Ck_vec = c(rep(0, k))
      
      for(i in 1:k){
        X_i = X_F[, -c(i+1)]
        SSE_i = t(y) %*% (In - X_i %*% solve(t(X_i) %*% X_i) %*% t(X_i)) %*% y
        C_ki = (SSE_i / MSE) + 2 * (k+1) - n
        Ck_vec[i] = C_ki
      }
      drop_index = which.min(Ck_vec)
    }
    ## 변수 제거 과정
    index_sol = c(drop_index) ; index_given = c(seq(from = 1, to = k, by = 1)) ; index_given = index_given[-index_sol]
    SS_beta = ASS_calc_char(X, y, index_vec = colname_vec,
                            index_sol = index_sol, index_given = index_given, coef = TRUE)
    
    F_alpha_drop = qf(alpha_drop, 1, n - k - 1, lower.tail = FALSE)
    F_L = SS_beta / MSE
    if (F_L < F_alpha_drop){
      X = X[, -drop_index]
      print(glue("제거된 설명변수 : {colname_vec[drop_index]}, F_L = {F_L} < F_alpha_drop = {F_alpha_drop}"))
      k = k - 1
    }else{
      end_operator = TRUE
      print(glue("{colname_vec[drop_index]} 는 제거할 수 없음, 후진제거법 종료 : F_L = {F_L} > F_alpha_drop = {F_alpha_drop}"))
      # break
    } 
    
  }
  final_var_vec = colnames(X)
  print(glue("최종 선택 변수 : {paste(final_var_vec, collapse = ', ')}"))
  
  return(list(X_out = X, final_var_vec = final_var_vec))
}


#######################################################################################
# 전진선택법

forward_sel = function(X, y, alpha_add = 0.15, method = c("MSE", "R2", "adj_R2", "Mallow")){
  library(glue)
  X = as.matrix(X) ; y = as.matrix(y)
  
  n = dim(X)[1] ; p = dim(X)[2]
  one = c(rep(1, n)) ; In = diag(1,n) ; Jn = one %*% t(one) ; 
  SST = t(y) %*% (In - (1/n)*Jn) %*% y
  SSE_full = t(y) %*% (In - X %*% solve(t(X) %*% X) %*% t(X) ) %*% y
  MSE_full = SSE_full / (n - p - 1)
  
  
  end_operator = FALSE
  
  # initialize
  X_temp = cbind(one) ; k = 1
  colname_vec = colnames(X) 
  
  while(end_operator == FALSE){
    
    num_var = length(colname_vec)
    MSE_vec = c(rep(0, num_var))
    
    if (method == "MSE"){
      for (i in 1:num_var){
        X_temp_null = X_temp
        X_temp_add = X[,colname_vec[i]]
        
        X_update = cbind(X_temp_null, X_temp_add)
        
        SSE = t(y) %*% (In - X_update %*% solve(t(X_update) %*% X_update) %*% t(X_update)) %*% y
        MSE = SSE / (n - k - 1)
        MSE_vec[i] = MSE
      }
      add_index = which.min(MSE_vec)
      print(glue("평균제곱오차(MSE) = {MSE_vec[add_index]} 를 가장 작게 하는 설명변수 : {colname_vec[add_index]}"))
    }else if(method == "R2"){
      R2_vec = c(rep(0, num_var))
      for (i in 1:num_var){
        X_temp_null = X_temp
        X_temp_add = X[,colname_vec[i]]
        
        X_update = cbind(X_temp_null, X_temp_add)
        
        SSE = t(y) %*% (In - X_update %*% solve(t(X_update) %*% X_update) %*% t(X_update)) %*% y
        R2 = 1 - (SSE/SST) ; MSE = SSE / (n - k - 1)
        R2_vec[i] = R2
        MSE_vec[i] = MSE
      }
      add_index = which.max(R2_vec)
      print(glue("결정계수(R2) = {R2_vec[add_index]} 를 가장 크게 하는 설명변수 : {colname_vec[add_index]}"))
    }else if(method == "adj_R2"){
      adj_R2_vec = c(rep(0, num_var))
      for (i in 1:num_var){
        X_temp_null = X_temp
        X_temp_add = X[,colname_vec[i]]
        
        X_update = cbind(X_temp_null, X_temp_add)
        
        SSE = t(y) %*% (In - X_update %*% solve(t(X_update) %*% X_update) %*% t(X_update)) %*% y
        R2 = 1 - (SSE/SST) ; MSE = SSE / (n - k - 1)
        adj_R2 = 1 - ((n - 1)/(n-k-1)) * (1 - R2)
        adj_R2_vec[i] = adj_R2
        MSE_vec[i] = MSE
      }
      add_index = which.max(adj_R2_vec)
      print(glue("수정결정계수(adj_R2) = {adj_R2_vec[add_index]} 를 가장 크게 하는 설명변수 : {colname_vec[add_index]}"))      
    }else if(method == "Mallow"){
      Ck_vec = c(rep(0, num_var))
      for (i in 1:num_var){
        X_temp_null = X_temp
        X_temp_add = X[,colname_vec[i]]
        
        X_update = cbind(X_temp_null, X_temp_add)
        
        SSE = t(y) %*% (In - X_update %*% solve(t(X_update) %*% X_update) %*% t(X_update)) %*% y
        MSE = SSE / (n - k - 1)
        MSE_vec[i] = MSE
        
        C_ki = (SSE / MSE_full) + 2 * (k+1) - n
        Ck_vec[i] = C_ki
      }
      add_index = which.min(Ck_vec)
      print(glue("맬로우C_k = {Ck_vec[add_index]} 를 가장 작게 하는 설명변수 : {colname_vec[add_index]}"))      
    }

    new_X = cbind(X[, colname_vec[add_index]]) ; colnames(new_X) = colname_vec[add_index]
    
    X_temp = cbind(X_temp, new_X) 
    temp_colname_vec = colnames(X_temp)
    temp_L = length(temp_colname_vec)
    
    index_sol = c(temp_L) ; index_given = c(seq(from = 1, to = temp_L, by = 1)) ; index_given = index_given[-index_sol]
    SS_beta = ASS_calc_char(X_temp, y, index_vec = temp_colname_vec,
                            index_sol = index_sol, index_given = index_given, coef = FALSE)
    MSE = MSE_vec[add_index]
    F_0 = SS_beta / MSE
    F_alpha_add = qf(alpha_add, 1, n - k - 1, lower.tail = FALSE)
    
    if (F_0 > F_alpha_add){
      print(glue("선택된 설명변수 : {colname_vec[add_index]}, F_0 = {F_0} > F_alpha_add = {F_alpha_add}"))
      k = k + 1
      colname_vec = colname_vec[-add_index]
      print(glue("\n"))
    }else{
      end_operator = TRUE
      del_index = dim(X_temp)[2]
      X_out = X_temp[,-c(1, del_index)]
      print(glue("{colname_vec[add_index]} 는 선택할 수 없음, 전진선택법 종료 : F_0 = {F_0} < F_alpha_add = {F_alpha_add}"))
      print(glue("\n"))
    }
    
  }
  final_var_vec = colnames(X_out)
  print(glue("최종 선택 변수 : {paste(final_var_vec, collapse = ', ')}"))
  return(list(X_out = X_out, final_var_vec = final_var_vec))
}






#######################################################################################
# 단계적 전진선택법

stepwise_sel = function(X, y, alpha_add = 0.15, alpha_drop = 0.15, method = c("MSE", "R2", "adj_R2", "Mallow")){
  library(glue)
  X = as.matrix(X) ; y = as.matrix(y)
  
  n = dim(X)[1] ; p = dim(X)[2]
  one = c(rep(1, n)) ; In = diag(1,n) ; Jn = one %*% t(one) ; 
  SST = t(y) %*% (In - (1/n)*Jn) %*% y
  SSE_full = t(y) %*% (In - X %*% solve(t(X) %*% X) %*% t(X) ) %*% y
  MSE_full = SSE_full / (n - p - 1)
  
  
  end_operator = FALSE
  
  # initialize
  X_temp = cbind(one) ; k = 1
  colname_vec = colnames(X) 
  
  while(end_operator == FALSE){
    
    num_var = length(colname_vec)
    MSE_vec = c(rep(0, num_var))
    
    if (method == "MSE"){
      for (i in 1:num_var){
        X_temp_null = X_temp
        X_temp_add = X[,colname_vec[i]]
        
        X_update = cbind(X_temp_null, X_temp_add)
        
        SSE = t(y) %*% (In - X_update %*% solve(t(X_update) %*% X_update) %*% t(X_update)) %*% y
        MSE = SSE / (n - k - 1)
        MSE_vec[i] = MSE
      }
      add_index = which.min(MSE_vec)
      print(glue("평균제곱오차(MSE) = {MSE_vec[add_index]} 를 가장 작게 하는 설명변수 : {colname_vec[add_index]}"))
    }else if(method == "R2"){
      R2_vec = c(rep(0, num_var))
      for (i in 1:num_var){
        X_temp_null = X_temp
        X_temp_add = X[,colname_vec[i]]
        
        X_update = cbind(X_temp_null, X_temp_add)
        
        SSE = t(y) %*% (In - X_update %*% solve(t(X_update) %*% X_update) %*% t(X_update)) %*% y
        R2 = 1 - (SSE/SST) ; MSE = SSE / (n - k - 1)
        R2_vec[i] = R2
        MSE_vec[i] = MSE
      }
      add_index = which.max(R2_vec)
      print(glue("결정계수(R2) = {R2_vec[add_index]} 를 가장 크게 하는 설명변수 : {colname_vec[add_index]}"))
    }else if(method == "adj_R2"){
      adj_R2_vec = c(rep(0, num_var))
      for (i in 1:num_var){
        X_temp_null = X_temp
        X_temp_add = X[,colname_vec[i]]
        
        X_update = cbind(X_temp_null, X_temp_add)
        
        SSE = t(y) %*% (In - X_update %*% solve(t(X_update) %*% X_update) %*% t(X_update)) %*% y
        R2 = 1 - (SSE/SST) ; MSE = SSE / (n - k - 1)
        adj_R2 = 1 - ((n - 1)/(n-k-1)) * (1 - R2)
        adj_R2_vec[i] = adj_R2
        MSE_vec[i] = MSE
      }
      add_index = which.max(adj_R2_vec)
      print(glue("수정결정계수(adj_R2) = {adj_R2_vec[add_index]} 를 가장 크게 하는 설명변수 : {colname_vec[add_index]}"))      
    }else if(method == "Mallow"){
      Ck_vec = c(rep(0, num_var))
      for (i in 1:num_var){
        X_temp_null = X_temp
        X_temp_add = X[,colname_vec[i]]
        
        X_update = cbind(X_temp_null, X_temp_add)
        
        SSE = t(y) %*% (In - X_update %*% solve(t(X_update) %*% X_update) %*% t(X_update)) %*% y
        MSE = SSE / (n - k - 1)
        MSE_vec[i] = MSE
        
        C_ki = (SSE / MSE_full) + 2 * (k+1) - n
        Ck_vec[i] = C_ki
      }
      add_index = which.min(Ck_vec)
      print(glue("맬로우C_k = {Ck_vec[add_index]} 를 가장 작게 하는 설명변수 : {colname_vec[add_index]}"))      
    }
    
    new_X = cbind(X[, colname_vec[add_index]]) ; colnames(new_X) = colname_vec[add_index]
    
    X_temp = cbind(X_temp, new_X) 
    temp_colname_vec = colnames(X_temp)
    temp_L = length(temp_colname_vec)
    
    index_sol = c(temp_L) ; index_given = c(seq(from = 1, to = temp_L, by = 1)) ; index_given = index_given[-index_sol]
    SS_beta = ASS_calc_char(X_temp, y, index_vec = temp_colname_vec,
                            index_sol = index_sol, index_given = index_given, coef = FALSE)
    MSE = MSE_vec[add_index]
    F_0 = SS_beta / MSE
    F_alpha_add = qf(alpha_add, 1, n - k - 1, lower.tail = FALSE)
    
    if (F_0 > F_alpha_add){
      print(glue("선택된 설명변수 : {colname_vec[add_index]}, F_0 = {F_0} > F_alpha_add = {F_alpha_add}"))
      k = k + 1
      colname_vec = colname_vec[-add_index]
      
      ## 두번째 루프로 넘어간 경우 : 이미 들어간 변수에 대해서도 유의성 검정하기
      if (dim(X_temp)[2] > 2){
        colname_vec_eli = colnames(X_temp)
        print(glue("설명변수 2개 이상 : {paste(colname_vec_eli[-1], collapse = ', ')} 후진제거법 시작"))
        
        SSE = t(y) %*% (In - X_temp %*% solve(t(X_temp) %*% X_temp) %*% t(X_temp)) %*% y
        MSE = SSE / (n - (dim(X_temp)[2]) )
        F_alpha_drop = qf(alpha_drop, 1, n - (dim(X_temp)[2]), lower.tail = FALSE)
        for (j in 2:(dim(X_temp)[2] - 1)){
          index_sol = c(j) ; index_given = c(seq(from = 1, to = dim(X_temp)[2], by = 1)) ; index_given = index_given[-index_sol]
          SS_beta = ASS_calc_char(X_temp, y, index_vec = temp_colname_vec,
                                  index_sol = index_sol, index_given = index_given, coef = FALSE)
          F_0 = SS_beta / MSE
          if (F_0 < F_alpha_drop){
            print(glue("제거해야 할 설명변수 : {colname_vec_eli[j]}, F_0 = {F_0} < F_alpha_drop = {F_alpha_drop}"))
            X_temp = X_temp[, -c(j)]
          }else {
            print(glue("설명변수 {colname_vec_eli[j]}는 유의함, F_0 = {F_0} > F_alpha_drop = {F_alpha_drop}"))
          }
        }
      }
      
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







#######################################################################################
# 단계적 후진제거법 # 4월 5일 이어서 하기

stepwise_eli = function(X, y, alpha_add = 0.15, alpha_drop = 0.15, method = c("MSE", "R2", "adj_R2", "Mallow")){
  library(glue)
  X = as.matrix(X)
  
  n = dim(X)[1] ; p = dim(X)[2]
  one = c(rep(1, n)) ; In = diag(1,n) ; Jn = one %*% t(one) ; 
  SST = t(y) %*% (In - (1/n)*Jn) %*% y
  X_eli = matrix(NA)
  
  
  end_operator = FALSE
  
  while(end_operator == FALSE){
    colname_vec = colnames(X)
    k = dim(X)[2] ; X_F = cbind(one, X)
    
    SSE = t(y) %*% (In - X_F %*% solve(t(X_F) %*% X_F) %*% t(X_F)) %*% y
    MSE = SSE / (n - k - 1)
    
    if (method == "MSE"){
      F_L_vec = c(rep(0, k))
      for(i in 1:k){
        index_sol = c(i) ; index_given = c(seq(from = 1, to = k, by = 1)) ; index_given = index_given[-index_sol]
        SS_beta = ASS_calc_char(X, y, index_vec = colname_vec,
                                index_sol = index_sol, index_given = index_given, coef = TRUE)
        F_L_element = SS_beta / MSE
        F_L_vec[i] = F_L_element
      }
      
      drop_index = which.min(F_L_vec)
      
    }else if (method == "R2"){
      R2_vec = c(rep(0, k))
      
      for(i in 1:k){
        X_i = X_F[, -c(i+1)]
        SSE_i = t(y) %*% (In - X_i %*% solve(t(X_i) %*% X_i) %*% t(X_i)) %*% y
        R2_i = 1 - (SSE_i / SST)
        R2_vec[i] = R2_i
      }
      drop_index = which.max(R2_vec)
      
    }else if(method == "adj_R2"){
      R2_a_vec = c(rep(0, k))
      
      for(i in 1:k){
        X_i = X_F[, -c(i+1)]
        SSE_i = t(y) %*% (In - X_i %*% solve(t(X_i) %*% X_i) %*% t(X_i)) %*% y
        R2_i = 1 - (SSE_i / SST)
        R2_ai = 1 - ((n - 1)/(n-k-1)) * (1 - R2_i)
        R2_a_vec[i] = R2_ai
      }
      drop_index = which.max(R2_a_vec)
      
    }else if(method == "Mallow"){
      Ck_vec = c(rep(0, k))
      
      for(i in 1:k){
        X_i = X_F[, -c(i+1)]
        SSE_i = t(y) %*% (In - X_i %*% solve(t(X_i) %*% X_i) %*% t(X_i)) %*% y
        C_ki = (SSE_i / MSE) + 2 * (k+1) - n
        Ck_vec[i] = C_ki
      }
      drop_index = which.min(Ck_vec)
    }
    ## 변수 제거 과정
    index_sol = c(drop_index) ; index_given = c(seq(from = 1, to = k, by = 1)) ; index_given = index_given[-index_sol]
    SS_beta = ASS_calc_char(X, y, index_vec = colname_vec,
                            index_sol = index_sol, index_given = index_given, coef = TRUE)
    
    F_alpha_drop = qf(alpha_drop, 1, n - k - 1, lower.tail = FALSE)
    F_L = SS_beta / MSE
    if (F_L < F_alpha_drop){
      X_eli = cbind(X_eli, X[, drop_index])
      X = X[, -drop_index]
      print(glue("제거된 설명변수 : {colname_vec[drop_index]}, F_L = {F_L} < F_alpha_drop = {F_alpha_drop}"))
      k = k - 1
      
      ## 두번째 루프로 넘어간 경우 : 이미 제거된 변수에 대해서도 유의성 검정하기
      if (dim(X_eli)[2] > 2){
        colname_vec_add = colnames(X_eli)
        print(glue("설명변수 2개 이상 : {paste(colname_vec_add[-1], collapse = ', ')} 전진선택법 시작"))
        
        SSE = t(y) %*% (In - X_eli %*% solve(t(X_eli) %*% X_eli) %*% t(X_eli)) %*% y
        MSE = SSE / (n - (dim(X_eli)[2]) )
        F_alpha_drop = qf(alpha_drop, 1, n - (dim(X_eli)[2]), lower.tail = FALSE)
        for (j in 2:(dim(X_eli)[2] - 1)){
          index_sol = c(j) ; index_given = c(seq(from = 1, to = dim(X_eli)[2], by = 1)) ; index_given = index_given[-index_sol]
          SS_beta = ASS_calc_char(X_eli, y, index_vec = temp_colname_vec,
                                  index_sol = index_sol, index_given = index_given, coef = FALSE)
          F_0 = SS_beta / MSE
          if (F_0 < F_alpha_add){
            print(glue("제거해야 할 설명변수 : {colname_vec_add[j]}, F_0 = {F_0} < F_alpha_add = {F_alpha_add}"))
            X_eli = X_eli[, -c(j)]
          }else {
            print(glue("설명변수 {colname_vec_add[j]}는 유의함, F_0 = {F_0} > F_alpha_add = {F_alpha_add}"))
            X = cbind(X, X_eli[, c(j)])
          }
        }
      }      
      
    }else{
      end_operator = TRUE
      print(glue("{colname_vec[drop_index]} 는 제거할 수 없음, 후진제거법 종료 : F_L = {F_L} > F_alpha_drop = {F_alpha_drop}"))
      # break
    } 
    
  }
  final_var_vec = colnames(X)
  print(glue("최종 선택 변수 : {paste(final_var_vec, collapse = ', ')}"))
  
  return(list(X_out = X, final_var_vec = final_var_vec))
}











#######################################################################################

# 전역 선택함수
best_subset_multi = function(X, y, method = "null"){
  library(glue)
  library(parallel)
  ncores <- detectCores() - 1
  cl <- makeCluster(ncores)
  
  X = as.matrix(X) ; y = as.matrix(y)
  
  n = dim(X)[1] ; p = dim(X)[2]
  one = c(rep(1, n)) ; In = diag(1,n) ; Jn = one %*% t(one)
  SST = t(y) %*% (In - (1/n)*Jn) %*% y
  SSE = t(y) %*% (In - X %*% solve(t(X) %*% X) %*% t(X)) %*% y
  MSE = SSE / (n - p - 1)
  
  colname_vec = colnames(X)
  
  colname_vec_res = c(total_subsets(colname_vec))
  
  # 각 부분집합의 원소들을 오름차순 정렬
  colname_vec_res = lapply(colname_vec_res, sort, decreasing = FALSE)
  
  # 각 부분집합의 길이(원소 개수)에 따라 전체 리스트 정렬 (0개, 1개, 2개, …)
  colname_vec_res_sorted = colname_vec_res[order(sapply(colname_vec_res, length))]
  
  colvec = colname_vec_res_sorted
  
  # 총 부분집합수
  K = length(colvec)
  
  # 중간계산 벡터
  SSE_vec = c(rep(0, K)) ; R2_a_vec = c(rep(0, K)) ; R2_vec = c(rep(0, K)) ; MSE_vec = c(rep(0, K)) ; C_k_vec = c(rep(0, K))
  
  # 결과물 벡터
  MSE_k_res = c(rep(0, p)) ; R2_k_res = c(rep(0, p))
  adj_R2_k_res = c(rep(0, p)) ; C_k_res = c(rep(0, p))
  
  
  
  min_MSE = Inf ; min_MSE_index = NA
  max_R2 = -1 ; max_R2_index = NA
  max_R2_a = -1 ; max_R2_a_index = NA
  min_C_k = Inf ; min_C_k_index = NA
  k = 1
  
  for (t in 1:K){
    subset_k = colvec[[t]]
    
    var_nums = length(subset_k)
    
    if (k < var_nums | k == p){
      MSE_k_res[k] = min_MSE
      R2_k_res[k] = max_R2
      adj_R2_k_res[k] = max_R2_a
      C_k_res[k] = min_C_k
      print(glue("k = {k} 일때 MSE기준 변수선택 : {paste(colvec[[min_MSE_index]], collapse = ', ')}, MSE = {MSE_k_res[k]}"))
      print(glue("k = {k} 일때 R2기준 변수선택 : {paste(colvec[[max_R2_index]],  collapse = ', ')}, R2 = {R2_k_res[k]}"))
      print(glue("k = {k} 일때 adj_R2기준 변수선택 : {paste(colvec[[max_R2_a_index]],  collapse = ', ')}, adj_R2 = {adj_R2_k_res[k]}"))
      print(glue("k = {k} 일때 C_k기준 변수선택 : {paste(colvec[[min_C_k_index]],  collapse = ', ')}, C_k = {C_k_res[k]}"))
      
      print(glue("\n"))
      k = k + 1
      print(glue("현재 k = {k}"))
    }
    
    X_k = X[, subset_k]
    X_k = cbind(one, X_k)
    X_ktX_k_inv = solve(t(X_k) %*% X_k)
    
    SSE_k = t(y) %*% (In - X_k %*% X_ktX_k_inv %*% t(X_k)) %*% y
    MSE_k = SSE_k / (n-k-1)
    R2_k = 1 - (SSE_k / SST)
    R2_ak = 1 - ((n-1)/(n-k-1)) * (1 - R2_k)
    C_k = (SSE_k / MSE) + 2*(k+1) - n
    
    SSE_vec[t] = SSE_k
    MSE_vec[t] = MSE_k
    R2_vec[t] = R2_k
    R2_a_vec[t] = R2_ak
    C_k_vec[t] = C_k
    
    if(MSE_vec[t] < min_MSE){
      min_MSE <- MSE_vec[t]
      min_MSE_index <- t
      # print(glue("최소의 MSE_{k} = {min_MSE} 를 가지는 {t}번째 부분집합 : {paste(subset_k, collapse = ', ')}"))
    }
    
    if(R2_vec[t] > max_R2){
      max_R2 <- R2_vec[t]
      max_R2_index <- t
      # print(glue("최대의 R2_{k} = {R2_k} 를 가지는 {t}번째 부분집합 : {paste(subset_k, collapse = ', ')}"))      
    }
    
    if(R2_a_vec[t] > max_R2_a){
      max_R2_a <- R2_a_vec[t]
      max_R2_a_index <- t
      # print(glue("최대의 R2_a{k} = {R2_ak} 를 가지는 {t}번째 부분집합 : {paste(subset_k, collapse = ', ')}"))
    }
    
    if(C_k_vec[t] < min_C_k){
      min_C_k =  C_k_vec[t]
      min_C_k_index = t
      
    }
    
    
    # print(glue("{t}번째 부분집합 : {paste(subset_k, collapse = ', ')}, SSE_{k} = {SSE_k}, R2_a{k} = {R2_ak} "))
    
  }
  MSE_k_res[k] = min_MSE
  R2_k_res[k] = max_R2
  adj_R2_k_res[k] = max_R2_a
  C_k_res[k] = min_C_k
  print(glue("k = {k} 일때 MSE기준 변수선택 : {paste(colvec[[min_MSE_index]], collapse = ', ')}, MSE = {MSE_k_res[k]}"))
  print(glue("k = {k} 일때 R2기준 변수선택 : {paste(colvec[[max_R2_index]],  collapse = ', ')}, R2 = {R2_k_res[k]}"))
  print(glue("k = {k} 일때 adj_R2기준 변수선택 : {paste(colvec[[max_R2_a_index]],  collapse = ', ')}, adj_R2 = {adj_R2_k_res[k]}"))
  print(glue("k = {k} 일때 C_k기준 변수선택 : {paste(colvec[[min_C_k_index]],  collapse = ', ')}, C_k = {C_k_res[k]}"))
  
  k = c(seq(from = 1, to = p, by = 1))
  table_out = data.frame(k, MSE_k_res, R2_k_res, adj_R2_k_res, C_k_res)
  colnames(table_out) = c("k", "MSE", "R2", "adj_R2", "C_k")
  
  plot(C_k_vec, type="b", xlab="Subset index", ylab="C_k")
  
  return(table_out)
}




#######################################################################################
# 단계적 전진선택법 (deprecated)

# stepwise_sel = function(X, y, alpha_add = 0.15, alpha_drop = 0.15){
#   library(glue)
#   X = as.matrix(X)
#   
#   n = dim(X)[1] ; p = dim(X)[2]
#   one = c(rep(1, n)) ; In = diag(1,n) ; Jn = one %*% t(one) ; 
#   SST = t(y) %*% (In - (1/n)*Jn) %*% y
#   
#   
#   end_operator = FALSE
#   
#   # initialize
#   X_temp = cbind(one) ; k = 1
#   colname_vec = colnames(X) 
#   
#   while(end_operator == FALSE){
#     
#     num_var = length(colname_vec)
#     R2_vec = c(rep(0, num_var))
#     MSE_vec = c(rep(0, num_var))
#     for (i in 1:num_var){
#       X_temp_null = X_temp
#       X_temp_add = X[,colname_vec[i]]
#       
#       X_update = cbind(X_temp_null, X_temp_add)
#       
#       SSE = t(y) %*% (In - X_update %*% solve(t(X_update) %*% X_update) %*% t(X_update)) %*% y
#       R2 = 1 - (SSE/SST) ; MSE = SSE / (n - k - 1)
#       R2_vec[i] = R2
#       MSE_vec[i] = MSE
#     }
#     
#     max_R2_index = which.max(R2_vec)
#     new_X = cbind(X[, colname_vec[max_R2_index]]) ; colnames(new_X) = colname_vec[max_R2_index]
#     print(glue("결정계수(R2) = {R2_vec[max_R2_index]} 를 가장 크게 하는 설명변수 : {colname_vec[max_R2_index]}"))
#     X_temp = cbind(X_temp, new_X) 
#     temp_colname_vec = colnames(X_temp)
#     temp_L = length(temp_colname_vec)
#     
#     index_sol = c(temp_L) ; index_given = c(seq(from = 1, to = temp_L, by = 1)) ; index_given = index_given[-index_sol]
#     SS_beta = ASS_calc_char(X_temp, y, index_vec = temp_colname_vec,
#                             index_sol = index_sol, index_given = index_given, coef = FALSE)
#     MSE = MSE_vec[max_R2_index]
#     F_0 = SS_beta / MSE
#     F_alpha_add = qf(alpha_add, 1, n - k - 1, lower.tail = FALSE)
#     
#     if (F_0 > F_alpha_add){
#       print(glue("선택된 설명변수 : {colname_vec[max_R2_index]}, F_0 = {F_0} > F_alpha_add = {F_alpha_add}"))
#       k = k + 1
#       colname_vec = colname_vec[-max_R2_index]
#       
#       ## 두번째 루프로 넘어간 경우 : 이미 들어간 변수에 대해서도 유의성 검정하기
#       if (dim(X_temp)[2] > 2){
#         colname_vec_eli = colnames(X_temp)
#         print(glue("설명변수 2개 이상 : {paste(colname_vec_eli[-1], collapse = ', ')} 후진제거법 시작"))
#         
#         SSE = t(y) %*% (In - X_temp %*% solve(t(X_temp) %*% X_temp) %*% t(X_temp)) %*% y
#         MSE = SSE / (n - (dim(X_temp)[2]) )
#         F_alpha_drop = qf(alpha_drop, 1, n - (dim(X_temp)[2]), lower.tail = FALSE)
#         for (j in 2:(dim(X_temp)[2] - 1)){
#           index_sol = c(j) ; index_given = c(seq(from = 1, to = dim(X_temp)[2], by = 1)) ; index_given = index_given[-index_sol]
#           SS_beta = ASS_calc_char(X_temp, y, index_vec = temp_colname_vec,
#                                   index_sol = index_sol, index_given = index_given, coef = FALSE)
#           F_0 = SS_beta / MSE
#           if (F_0 < F_alpha_drop){
#             print(glue("제거해야 할 설명변수 : {colname_vec_eli[j]}, F_0 = {F_0} < F_alpha_drop = {F_alpha_drop}"))
#             X_temp = X_temp[, -c(j)]
#           }else {
#             print(glue("설명변수 {colname_vec_eli[j]}는 유의함, F_0 = {F_0} > F_alpha_drop = {F_alpha_drop}"))
#           }
#         }
#       }
#       print(glue("\n"))
#     }else{
#       end_operator = TRUE
#       del_index = dim(X_temp)[2]
#       X_out = X_temp[,-c(1, del_index)]
#       print(glue("더 이상 변수를 선택할 수 없음, 전진선택법 종료 : F_0 = {F_0} < F_alpha_add = {F_alpha_add}"))
#       print(glue("\n"))
#     }
#     
#   }
#   final_var_vec = colnames(X_out)
#   print(glue("최종 선택 변수 : {paste(final_var_vec, collapse = ', ')}"))
#   return(list(X_out = X_out, final_var_vec = final_var_vec))
# }