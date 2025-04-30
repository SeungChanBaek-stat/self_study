source("functions/mult_reg.R", echo = F)


#######################################################################################

# 능형회귀 (Ridge Regression)
ridge_reg = function(X, y, lambda){
  library(glue)
  rowname_vec_X = colnames(X) ; colname_vec_X = as.character(lambda) ; name_vec_y = names(y)
  X = as.matrix(X) ; y = as.matrix(y)
  
  ## 표준화
  n = dim(X)[1] ; p = dim(X)[2] ; L = length(lambda) ; I = diag(1, p)
  
  ### 1. 8장 3절에서 소개하는 표준화방식
  # standard_res = standard_calc(X, y)
  # X = standard_res$Z ; y = standard_res$ystar
  
  ### 2. 중심화 행렬 C(centering matrix) 방식
  # In = diag(1, n) ; one = c(rep(1,n)) ; C = In - (1/n) * one %*% t(one)
  # Xc = C %*% X ; yc = C %*% y
  # Sx = t(Xc) %*% Xc ; Sy = as.numeric( t(yc) %*% yc )
  # sdX = sqrt( diag(Sx) ) ; Dx = diag(sdX) ; sdy = sqrt( Sy )
  # X = Xc %*% solve(Dx) ; y = yc / sdy
  
  ### 3. scale 함수 이용
  X = scale(X, center = TRUE, scale = TRUE)
  y = as.numeric(scale(y, center = TRUE, scale = TRUE))
  ## 능형회귀추정
  beta_hat_ridge = matrix(NA, nrow = p, ncol = L)
  
  for (j in 1:L){
    k = lambda[j]
    XtX = t(X) %*% X ; kI = k * I 
    beta_hat = solve(XtX + kI) %*% t(X) %*% y
    beta_hat_ridge[,j] = beta_hat
  }
  
  rownames(beta_hat_ridge) = rowname_vec_X
  colnames(beta_hat_ridge) = colname_vec_X
  
  return(beta_hat_ridge)
}