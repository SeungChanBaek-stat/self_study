#######################################################################################

total_subsets <- function(set) { 
  n <- length(set)
  masks <- 2^(1:n-1)
  res = sapply( 1:2^n-1, function(u) set[ bitwAnd(u, masks) != 0 ] )
  
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
  
  return(list(col_vec = colvec, MSE_vec = MSE_vec, R2_a_vec = R2_a_vec, min_MSE_index, max_R2_a_index))
}  
  
