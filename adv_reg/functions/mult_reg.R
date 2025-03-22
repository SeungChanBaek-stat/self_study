## 중회귀에서의 분산분석용 함수



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
  if (coeff == TRUE){
    n = length(y) ; p = dim(X)[2] - 1
    one = c(rep(1,n)) ; In = diag(1, n) ; Jn_n = one %*% t(one) / n
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
    n = length(y) ; p = dim(X)[2]
    In = diag(1, n)
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