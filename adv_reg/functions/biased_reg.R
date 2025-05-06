source("functions/mult_reg.R", echo = F)


#######################################################################################

# 능형회귀 (Ridge Regression)
ridge_reg = function(X, y, lambda){
  library(glue)
  rowname_vec_X = colnames(X) ; colname_vec_X = as.character(lambda) ; name_vec_y = names(y)
  # X = as.matrix(X) ; y = as.matrix(y)
  
  ## 표준화
  n = dim(X)[1] ; p = dim(X)[2] ; L = length(lambda) ; Ip = diag(1, p) ; In = diag(1, n)
  
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
  # y = as.numeric(scale(y, center = TRUE, scale = TRUE))
  y = scale(y, center = TRUE, scale = TRUE)
  
  X = as.matrix(X) ; y = as.matrix(y)
  
  ## 능형회귀추정
  beta_hat_ridge = matrix(NA, nrow = p, ncol = L)
  
  if (p >= n){
    print(glue("High dimensional problem : p = {p} > n = {n}"))
    X_svd <- svd(X, nu = n, nv = p)
    U = X_svd$u ; d = X_svd$d ; V = X_svd$v
    V_pn = V[,c(1:n)] ; D_nn = diag(d)
    
    for (j in 1:L){
      k = lambda[j]
      DtD = t(D_nn) %*% D_nn ; kIn = k * In
      beta_hat = V_pn %*% solve(DtD + kIn) %*% t(D_nn) %*% t(U) %*% y
      beta_hat_ridge[,j] = beta_hat
    }
    
  }else{
    for (j in 1:L){
      k = lambda[j]
      XtX = t(X) %*% X ; kIp = k * Ip 
      beta_hat = solve(XtX + kIp) %*% t(X) %*% y
      beta_hat_ridge[,j] = beta_hat
    }    
  }
  

  
  rownames(beta_hat_ridge) = rowname_vec_X
  colnames(beta_hat_ridge) = colname_vec_X
  
  return(beta_hat_ridge)
}


#######################################################################################

# 주성분회귀 (Principal Component Regression)
pc_reg = function(X, y, alpha = 0.05){
  ## X 와 y는 이미 표준화가 되어있다고 가정, as.matrix까지 된 상태로 가정한다.
  library(glue)
  n = dim(X)[1] ; p = dim(X)[2]
  eigen_decomp = eigen(cor(X))
  D = diag(eigen_decomp$values) ; P = eigen_decomp$vectors
  Z = X %*% P
  pcr = mult_reg(Z, y, alpha = alpha, coeff = TRUE)
  plot(eigen_decomp$values, type = "l", xlab = "No. of components",
       ylab = expression(lambda), main = "Scree Plot for PCR")
  
  return(pcr)
}



#######################################################################################

# 부분최소제곱회귀 (Partial Least Squares Regression)
pls_reg = function(X, y, alpha = 0.05, method = "ortho_invar"){
  library(glue)
  library(MASS)
  ## 표준화
  n = dim(X)[1] ; p = dim(X)[2] ; In = diag(1,n)
  X_mat <- as.matrix(X)
  y_mat <- as.matrix(y)
  
  V1 <- sweep(X_mat, 2, apply(X_mat, 2, mean))
  U1 <- y_mat - mean(y_mat)
  Cy = U1
  
  ### U, V, T 초기화
  U_mat = cbind() ; V_mat = cbind() ; T_mat = cbind()
  
  for (j in 1:p){
    ## 초기값 (V1 은 X 표준화, U1은 y 표준화)
    if (j == 1){
      V1 <- sweep(X_mat, 2, apply(X_mat, 2, mean))
      U1 <- y_mat - mean(y_mat)
      w1 = c(rep(0, p))
      
      ## 가중치 설정
      if (method == "ortho_invar"){
        for (k in 1:p){
          w1[k] = crossprod(V1[,k], V1[,k])
        }
      }else if (method == "scale_invar"){
        w1[k] = 1/p
      }
      
      ## phi_hat 계산
      phi1_hat = c(rep(0, p))
      for (k in 1:p){
        phi1_hat[k] = crossprod(V1[,k],U1) / crossprod(V1[,k], V1[,k])
      }
      
      T1 = V1 %*% (phi1_hat * w1)
      U_mat = cbind(U1) ; V_mat = cbind(V1) ; T_mat = cbind(T1)
      U_old = U1 ; V_old = V1 ; T_old = T1
      colnames(T_mat)[ncol(T_mat)] <- paste0("T", 1)
      
      ## 최소제곱회귀를 통한 결정계수 계산
      beta_hat = qr.solve(T_mat, Cy)
      # beta_hat <- ginv(crossprod(T_mat)) %*% crossprod(T_mat, Cy)
      SSR = t(beta_hat) %*% t(T_mat) %*% Cy
      SST = t(Cy) %*% (In) %*% Cy
      R2 = SSR / SST
      print(glue("T{j} 에 대한 회귀모형의 결정계수 = {R2} "))
    ## l = 2, ..., p-1 case
    }else{
      V_new = matrix(NA, nrow = n, ncol = p)
      w_new = c(rep(0, p))
      # print(dim(T_old))
      for (k in 1:p){
        # print((crossprod(T_old, V_old[,k]) / crossprod(T_old, T_old)))
        # print(dim((crossprod(T_old, V_old[,k]) / crossprod(T_old, T_old))))
        V_new[,k] = V_old[,k] - as.numeric(crossprod(T_old, V_old[,k]) / crossprod(T_old, T_old)) * T_old
        U_new = U_old - as.numeric(crossprod(T_old, V_old[,k]) / crossprod(T_old, T_old)) * T_old
      }
      
      ## 가중치 설정
      if (method == "ortho_invar"){
        for (k in 1:p){
          w_new[k] = crossprod(V_new[,k], V_new[,k])
        }
      }else if (method == "scale_invar"){
        w_new[k] = 1/p
      }
      
      ## phi_hat 계산
      phi_hat_new = c(rep(0, p))
      for (k in 1:p){
        phi_hat_new[k] = crossprod(V_new[,k],U_new) / crossprod(V_new[,k], V_new[,k])
      }
      T_new = V_new %*% (phi_hat_new * w_new)
      
      ## 열 저장
      U_mat = cbind(U_mat, U_new) ; V_mat = cbind(V_mat, V_new) ; T_mat = cbind(T_mat, T_new)
      colnames(T_mat)[ncol(T_mat)] <- paste0("T", j)
      
      ## 최소제곱회귀를 통한 결정계수 계산
      beta_hat = qr.solve(T_mat, Cy)
      # beta_hat <- ginv(crossprod(T_mat)) %*% crossprod(T_mat, Cy)
      SSR = t(beta_hat) %*% t(T_mat) %*% Cy
      SST = t(Cy) %*% (In) %*% Cy
      R2 = SSR / SST
      print(glue("T1, ..., T{j} 에 대한 회귀모형의 결정계수 = {R2} "))
      
      ## 변수 재설정
      U_old = U_new ; V_old = V_new ; T_old = T_new
      
    }
  }
  return(list(T_mat = T_mat, Cy = Cy))
}