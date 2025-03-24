


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
              MSR = MSR, MSE = MSE))
}









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