
#######################################################################################

# 중회귀에서의 분산분석용 함수
mult_reg = function(X, y, alpha = 0.05, coeff = TRUE){
  library(glue)
  
  # X가 행렬인지 확인
  if (!is.matrix(X)) {
    stop("X는 행렬이어야 합니다.")
  }
  
  # y가 숫자형 벡터 혹은 행렬인지 확인
  if (!is.numeric(y)) {
    stop("y는 숫자형이어야 합니다.")
  }
  
  n = length(y) ; p = dim(X)[2]
  In = diag(1, n)
  
  if (coeff == TRUE){
    one = c(rep(1,n)) ; Jn_n = one %*% t(one) / n
    X = cbind(one, X)
    H = X %*% solve(t(X) %*% X) %*% t(X)
    XtX = t(X) %*% X ; Xty = t(X) %*% y
    beta_hat = solve(XtX) %*% Xty
    
    SST = t(y) %*% (In - Jn_n) %*% y
    SSE = t(y) %*% (In - H) %*% y
    SSR = t(y) %*% (H - Jn_n) %*% y
    
    df_SSR = p ; df_SSE = n - p - 1
    
    MSR = SSR / df_SSR ; MSE = SSE/ df_SSE
    
    F_0 = MSR/MSE ; F_alpha = qf(alpha, p, n-p-1, lower.tail = FALSE)
    
  }else{
    H = X %*% solve(t(X) %*% X) %*% t(X)
    XtX = t(X) %*% X ; Xty = t(X) %*% y
    beta_hat = solve(XtX) %*% Xty
    
    SST = t(y) %*% (In) %*% y
    SSE = t(y) %*% (In - H) %*% y
    SSR = t(y) %*% (H) %*% y
    
    df_SSR = p ; df_SSE = n - p
    
    MSR = SSR / df_SSR ; MSE = SSE/ df_SSE
    
    F_0 = MSR/MSE ; F_alpha = qf(alpha, p, n-p, lower.tail = FALSE)
    
  }

  anova_table <- data.frame(
    요인 = c("회귀", "잔차", "계"),
    제곱합 = c(SSR, SSE, SST),
    자유도 = c(df_SSR, df_SSE, df_SSR + df_SSE),
    평균제곱합 = c(MSR, MSE, NA),
    F값 = c(F_0, NA, NA),
    F기각치 = c(F_alpha, NA, NA)
  )
  SST = as.numeric(SST) ; SSR = as.numeric(SSR) ; SSE = as.numeric(SSE)
  MSR = as.numeric(MSR) ; MSE = as.numeric(MSE)
  
  print(anova_table)

  print(glue("SST = {SST}, SSR = {SSR}, SSE = {SSE}"))
  
  print(glue("MSR = {MSR}, MSE = {MSE}"))
  
  print(glue("F_0 = {F_0}, F_alpha = {F_alpha}"))
  
  return(list(XtX = XtX, Xty = Xty, beta_hat = beta_hat,
              SST = SST, SSR = SSR, SSE = SSE,
              MSR = MSR, MSE = MSE, anova_table = anova_table))
}





#######################################################################################



# CB = m 검정용 함수
mult_test = function(C, m, X, X_r, y, y_r, alpha = 0.05, method = "one", coef = TRUE){
  
  ## n, p, In, k 설정정
  n = dim(X)[1] ; p = dim(X)[2] ; k = dim(C)[1] ; In = diag(1, n)
  
  ## 방법 1, 절편이 있는 모형
  if (coef == TRUE & method == "one"){
    
    one = c(rep(1, n)) ; X = cbind(one, X)
    
    XtX_inv = solve(t(X) %*% X)
    H = X %*% XtX_inv %*% t(X)
    
    beta_hat = XtX_inv %*% t(X) %*% y
    
    SSE = t(y) %*% (In - H) %*% y
    Cb_m = C %*% beta_hat - m
    
    Q = t(Cb_m) %*% solve(C %*% XtX_inv %*% t(C)) %*% (Cb_m)
    
    F_0 = (Q/k) / (SSE / (n-p-1)) ; F_alpha = qf(alpha, k, n-p-1, lower.tail = FALSE)
    print(glue("F_0 = {F_0}, F_alpha = {F_alpha}"))
    
    return(list(Q = Q, SSE = SSE, F_0 = F_0, F_alpha = F_alpha))

    
    
    ## 방법 1, 절편이 없는 모형
  }else if (coef == FALSE & method == "one"){
    
    XtX_inv = solve(t(X) %*% X)
    H = X %*% XtX_inv %*% t(X)
    
    beta_hat = XtX_inv %*% t(X) %*% y
    
    SSE = t(y) %*% (In - H) %*% y
    Cb_m = C %*% beta_hat - m
    
    Q = t(Cb_m) %*% solve(C %*% XtX_inv %*% t(C)) %*% (Cb_m)
    
    F_0 = (Q/k) / (SSE / (n - p)) ; F_alpha = qf(alpha, k, n-p, lower.tail = FALSE)
    print(glue("F_0 = {F_0}, F_alpha = {F_alpha}"))
    
    return(list(Q = Q, SSE = SSE, F_0 = F_0, F_alpha = F_alpha))
    
    
    
    ## 방법 2, 절편이 있는 모형  
  }else if (coef == TRUE & method == "two"){
    
    one = c(rep(1, n)) ; X = cbind(one, X) ; X_r = cbind(one, X_r)
    
    XtX_inv = solve(t(X) %*% X)
    H = X %*% XtX_inv %*% t(X)
    
    X_rtX_r_inv = solve(t(X_r) %*% X_r)
    H_r = X_r %*% X_rtX_r_inv %*% t(X_r)
    
    beta_hat = XtX_inv %*% t(X) %*% y
    
    SSE_F = t(y) %*% (In - H) %*% y
    SSE_R = t(y_r) %*% (In - H_r) %*% y_r
    
    F_0 = (SSE_R - SSE_F)/(k * SSE_F / (n-p-1)) ; F_alpha = qf(alpha, k, n-p-1, lower.tail = FALSE)
    print(glue("F_0 = {F_0}, F_alpha = {F_alpha}"))
    
    return(list(SSE_F = SSE_F, SSE_R = SSE_R, F_0 = F_0, F_alpha = F_alpha))
    
    
    
    ## 방법 2, 절편이 없는 모형    
  }else if (coef == FALSE & method == "two"){
    
    XtX_inv = solve(t(X) %*% X)
    H = X %*% XtX_inv %*% t(X)
    
    X_rtX_r_inv = solve(t(X_r) %*% X_r)
    H_r = X_r %*% X_rtX_r_inv %*% t(X_r)
    
    beta_hat = XtX_inv %*% t(X) %*% y
    
    one = c(rep(1,n)) ; In = diag(1, n) ; Jn_n = one %*% t(one) / n
    
    SSE_F = t(y) %*% (In - H) %*% y
    SSE_R = t(y_r) %*% (In - H_r) %*% y_r
    SSR_R = t(y_r) %*% (H_r) %*% y_r
    SST_R = t(y_r) %*% y_r
    
    print(glue("SSE_R = {SSE_R}, SSE_F = {SSE_F}, SSR_R = {SSR_R}, SST_R = {SST_R}, SSE_R + SSR_R = {SSE_R + SSR_R}"))
    F_0 = (SSE_R - SSE_F)/(k * SSE_F / (n-p)) ; F_alpha = qf(alpha, k, n-p, lower.tail = FALSE)
    print(glue("F_0 = {F_0}, F_alpha = {F_alpha}"))
    
    return(list(SSE_F = SSE_F, SSE_R = SSE_R, F_0 = F_0, F_alpha = F_alpha))
  }
}

#######################################################################################

# 추가제곱합 계산함수
ASS_calc = function(X, y, index_sol, index_given = NA, coef = TRUE){
  n = dim(X)[1] ; p = dim(X)[2]
  if (coef == TRUE){
    one = c(rep(1,n)) ; X = cbind(one, X)
  }
  if (all(is.na(index_given))){
    index_sol = index_sol + 1
    X_sol = X[,index_sol]
    SS = t(y) %*% X_sol %*% solve(t(X_sol) %*% X_sol) %*% t(X_sol) %*% y
    
    print(glue("SS({paste0('beta_', index_sol - 1, collapse = ', ')}) = {SS}"))
    return(SS)
    
  }else{
    index_sol = index_sol + 1 ; index_given = index_given + 1
    index_full <- sort(union(index_sol, index_given))
    X_F = X[,index_full] ; X_G = X[,index_given]
    SS_F = t(y) %*% X_F %*% solve(t(X_F) %*% X_F) %*% t(X_F) %*% y
    SS_G = t(y) %*% X_G %*% solve(t(X_G) %*% X_G) %*% t(X_G) %*% y
    
    SS = SS_F - SS_G
    
    # print(glue("SS(beta_{index_sol - 1} | beta_{paste(index_given - 1, collapse = ', ')}) = {SS}"))
    print(glue("SS({paste0('beta_', index_sol - 1, collapse = ', ')} | {paste0('beta_', index_given - 1, collapse = ', ')}) = {SS}"))
    return(SS)
  }
}



#######################################################################################

# 추가제곱합 계산함수2(변수명 character를 다루는 경우)
ASS_calc_char = function(X, y, index_vec, index_sol, index_given = NA, coef = TRUE){
  n = dim(X)[1] ; p = dim(X)[2]
  if (coef == TRUE){
    one = c(rep(1,n)) ; X = cbind(one, X)
  }
  if (all(is.na(index_given))){
    index_sol = index_sol + 1
    X_sol = X[,index_vec[index_sol]]
    SS = t(y) %*% X_sol %*% solve(t(X_sol) %*% X_sol) %*% t(X_sol) %*% y
    
    print(glue("SS({paste0('beta_', index_sol - 1, collapse = ', ')}) = {SS}"))
    return(SS)
    
  }else{
    if (coef == TRUE){
      index_sol = index_sol + 1 ; index_sol = c(index_sol)
      index_given = index_given + 1 ; index_given = c(1, index_given)
      index_full <- sort(union(index_sol, index_given))
      
      
    }else if(coef == FALSE){
      index_full <- sort(union(index_sol, index_given))
    }
    # X_F = X[,index_vec[index_full]] ; X_G = X[,index_vec[index_given]]
    X_F = X[,index_full] ; X_G = X[,index_given]
    SS_F = t(y) %*% X_F %*% solve(t(X_F) %*% X_F) %*% t(X_F) %*% y
    SS_G = t(y) %*% X_G %*% solve(t(X_G) %*% X_G) %*% t(X_G) %*% y
    
    SS = SS_F - SS_G
    
    # print(glue("SS({paste0('beta_', index_sol - 1, collapse = ', ')} | {paste0('beta_', index_given - 1, collapse = ', ')}) = {SS}"))
    return(SS)
    
  }
}


#######################################################################################

# 표준화함수
standard_calc = function(X, y){
  n = dim(X)[1] ; p = dim(X)[2]
  
  xbar = c(rep(0, p))
  for (j in 1:p){
    xbar[j] = mean(X[,j])
  }
  
  S_jj = c(rep(0, p))
  for (j in 1:p){
    S_jj[j] = sum((X[,j] - xbar[j])^2)
  }
  
  ybar = mean(y) ; S_yy = sum((y - ybar)^2)
  
  Z = matrix(NA, nrow = n, ncol = p)
  for (j in 1:p){
    Z[,j] =  (X[,j] - xbar[j]) / sqrt(S_jj[j])
  }
  
  ystar = (y - ybar) / sqrt(S_yy)
  
  return(list(Z = Z, ystar = ystar, xbar = xbar, ybar = ybar, S_jj = S_jj, S_yy = S_yy))
}



#######################################################################################

# 직교다항회귀행렬 계산함수
ortho_poly = function(X, k, coef = TRUE){
  library(glue)
  n = dim(X)[1] ; p = dim(X)[2] ; d = X[2,1] - X[1,1]
  
  ## 설명변수의 간격 체크
  for (i in 1:(n-1)){
    temp_d = X[i+1,1] - X[i,1]
    if (temp_d != d){
      cat("설명변수 x의 수준이 같은 간격으로 떨어져있지 않습니다.")
      return( )
    }
  }
  
  ## 절편항 추가
  if (coef == TRUE){    one = c(rep(1, n)) ; X = cbind(one, X)  }
  
  ## 직교다항행렬 X_p 생성
  X_p = matrix(NA, nrow = n, ncol = k + 1)
  
  xbar = mean(X[,2])
  X_p[,1] = 1
  X_p[,2] = (X[,2] - xbar)/d
  
  if (k == 1){
    colnames(X_p)= c("one","x^1")
    return(X_p)
  }else{
    colname_vec = c(rep("1", k+1))
    colname_vec[1] = "one"
    
    for (r in 2:k){
      X_p[ ,r+1] = X_p[ ,r] * X_p[ ,2] - ((r-1)^2 * (n^2 - (r-1)^2))/(4 * (4*(r-1)^2 - 1)) * X_p[,r-1] 
      colname_vec[r] = glue("x^{r-1}")
    }
    colname_vec[k+1] = glue("x^{k}")
    
    colnames(X_p)= colname_vec
    return(X_p)
  }
}



#######################################################################################

# 가변수 생성함수
dummy_var_gen = function(X){
  library(glue)
  n = dim(X)[1] ; p = dim(X)[2] ; colnamevec = colnames(X)
  uniq_name_vec = c()
  print(glue("초기값 설정 완료"))
  
  for (j in 1:p){
    col_name = colnamevec[j]
    # print(glue("{col_name} 연산 시작"))
    uniq_ = unique(X[,col_name])
    uniq_vec = as.vector(uniq_)
    # print(uniq_vec)
    if (class(uniq_) == "factor"){
      uniq_name_vec = c(uniq_name_vec, col_name)
      num_cat = length(uniq_vec)
      # print(glue("{col_name}의 종류 갯수 = {num_cat}"))
      for (k in 1:(num_cat - 1)){
        new_vec_name = paste(col_name, uniq_vec[k], sep="")
        # print(glue(new_vec_name))
        X[,new_vec_name] = ifelse(X[,col_name] == uniq_vec[k], 1, 0)
        # print(X$new_vec_name)
      }
    }
  }
  
  for (item in uniq_name_vec){
    X = X %>%
      dplyr::select(-all_of(item))
    # print(item)
  }
  X_dummy = X
  # X_dummy_vec = 
  print(glue("더미변수열 {paste(uniq_name_vec, collapse=", ")} 생성 완료"))
  # return(list(X = X_dummy, dummy_vec = X_dummy_vec))
  return(X_dummy)
}



#######################################################################################

# IRWLS 로지스틱 회귀함수
irwls_logistic_reg = function(X, y, tol = 1e-8, alpha = 0.05){
  library(glue)
  n = dim(X)[1] ; one = c(rep(1, n))
  X = cbind(one, X) ; p = dim(X)[2]
  tol = tol
  end_operator = FALSE
  
  ## 초기화
  pi_beta = c(rep(0, n))
  beta_old = c(rep(0, p))
  W_beta_vec = c(rep(0, n))
  iteration = 0
  
  while (end_operator == FALSE){
    for (i in 1:n){
      pi_beta[i] = exp(t(X[i,]) %*% beta_old ) / (1 + exp(t(X[i,])%*% beta_old ))
      W_beta_vec[i] = pi_beta[i] * (1 - pi_beta[i])
    }
    W_beta = diag(W_beta_vec)
    
    beta_new = beta_old + solve(t(X) %*% W_beta %*% X) %*% t(X) %*% (y - pi_beta)
    iteration = iteration + 1
    if (max(abs(beta_new - beta_old)) < tol){
      print(glue("converged at {iteration} iterations"))
      print(glue("supreme norm distance = {max(abs(beta_new - beta_old))}"))
      print(beta_new)
      end_operator = TRUE
    }else if (max(abs(beta_new - beta_old)) >= tol){
      # if (iteration %% 1000 == 0){
      #   print(glue("computing at {iteration} iterations"))
      #   print(glue("supreme norm distance = {max(abs(beta_new - beta_old))}"))
      # }
      print(glue("computing at {iteration} iterations"))
      print(glue("supreme norm distance = {max(abs(beta_new - beta_old))}"))
      beta_old = beta_new
    }
  }
  pi_beta = c(rep(0, n))
  W_beta_vec = c(rep(0, n))
  for (i in 1:n){
    pi_beta[i] = exp(t(X[i,]) %*% beta_new) / (1 + exp(t(X[i,]) %*% beta_new))
    W_beta_vec[i] = pi_beta[i] * (1 - pi_beta[i])
  }
  W_beta = diag(W_beta_vec)
  
  ### deprecated : 이 방식은 OLS 에서만 사용가능한 방법
  # In = diag(1, n) ; Jn_n = one %*% t(one) / n
  # SST = t(y) %*% (In - Jn_n) %*% y
  # # SST = t(y) %*% y
  # SSR = t(beta_new) %*% t(X) %*% y - t(y) %*% Jn_n %*% y
  # # SSR = t(beta_new) %*% t(X) %*% y - t(y) %*% y  
  # SSE = SST - SSR
  # 
  # df_SSR = p - 1 ; df_SSE = n - p
  # 
  # MSR = SSR / df_SSR ; MSE = SSE/ df_SSE
  # 
  # F_0 = MSR/MSE ; F_alpha = qf(alpha, df_SSR, df_SSE, lower.tail = FALSE)
  # 
  # anova_table <- data.frame(
  #   요인 = c("회귀", "잔차", "계"),
  #   제곱합 = c(SSR, SSE, SST),
  #   자유도 = c(df_SSR, df_SSE, df_SSR + df_SSE),
  #   평균제곱합 = c(MSR, MSE, NA),
  #   F값 = c(F_0, NA, NA),
  #   F기각치 = c(F_alpha, NA, NA)
  # )
  
  return(list(beta_hat = beta_new, W = W_beta))
}
