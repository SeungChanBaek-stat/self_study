
#######################################################################################

# 일원배치법
oneway_anova <- function(y, A_num = 1, repeat_vec = c(1), method = "null", alpha = 0.05){
  ## 반복수가 같은 경우나 다른 경우나, 어차피 R코드상의 계산방식에는 변함이 없다.
  ## 따라서 반복수에 대한 구분을 굳이 할 필요가 없다 -> diff_repeat 변수 삭제
  library(glue)
  ## 기본적인 X 행렬
  n = length(y) ; one = c(rep(1, n))
  X = cbind(one) ; ybar = mean(y)
  
  counter = 1
  for (i in 1:A_num){
    temp_vec = c(rep(0, n))
    R_num_A = repeat_vec[i]
    for (j in (counter) : (counter + R_num_A - 1)){
      temp_vec[j] = 1
    }
    X = cbind(X, temp_vec)
    counter = counter + R_num_A
  }
  colnames(X) = c("one", paste0("A_", 1:A_num))
  
  ## 모수재조정법 : 방법1 - 모형에 가정을 직접 대입하는 방식
  if (method == "one"){
    W = cbind(one)
    for (i in 1:(A_num - 1)){
      temp_W = X[,c(i+1)] - (repeat_vec[i]/repeat_vec[A_num]) * X[,c(A_num + 1)]
      W = cbind(W, temp_W)
      colnames(W)[ncol(W)] <- paste0("W_", i)
    }
    W = W[,-c(1)]
    
    mult_reg_res = mult_reg(W, y, alpha = alpha, coeff = TRUE)
    alpha_l_hat = -sum(mult_reg_res$beta_hat[2:A_num])
    alpha_hat = rbind(mult_reg_res$beta_hat, alpha_l_hat)
    alpha_hat_namevec = c(rep(0, A_num))
    for (i in 1:A_num){
      alpha_hat_namevec[i] = paste0("A_", i)
    }
    
    rownames(alpha_hat)[2:(A_num + 1)] = alpha_hat_namevec
    
    anova_table = mult_reg_res$anova_table
    
    return(list(X = W, alpha_hat = alpha_hat, anova_table = anova_table))    
  
  ## 모수재조정법 : 방법2 - 변수 1개를 제거하는 방식  
  }else if (method == "two"){
    X = X[,-c(A_num + 1)] ; X = X[,-c(1)]
    mult_reg_res = mult_reg(X, y, alpha = alpha, coeff = TRUE)
    
    gamma_hat = mult_reg_res$beta_hat
    SST = mult_reg_res$SST
    SSR = mult_reg_res$SSR
    SSE = mult_reg_res$SSE
    anova_table = mult_reg_res$anova_table
    
    return(list(X = X, gamma_hat = gamma_hat, anova_table = anova_table))   
    
  ## 정규방정식을 활용한 방법 : 가정을 이용한 모수추정  
  }else if (method == "null"){
    XtX = t(X) %*% X ; Xty = t(X) %*% y
    alpha_hat = matrix(NA, nrow = (A_num + 1))
    
    for (i in 1: (A_num + 1)){
      if (i == 1){
        alpha_hat[i, ] = Xty[i,]/XtX[1,i]
      }else{
        alpha_hat[i,] = Xty[i,]/XtX[1,i] - alpha_hat[1, ]
      }
    }
    
    alpha_hat_namevec = c(rep(0, A_num))
    
    for (i in 1:(A_num + 1)){
      if (i == 1){
        alpha_hat_namevec[i] = "one"
      }else{
        alpha_hat_namevec[i] = paste0("A_", (i-1))
      }
    }
    
    rownames(alpha_hat) = alpha_hat_namevec
    
    SSR = t(alpha_hat) %*% Xty - n * (ybar)^2
    SST = t(y) %*% y - n * (ybar)^2
    SSE = SST - SSR
    df_A = (A_num - 1) ; df_E = (n - A_num) ; df_T = n - 1
    V_A = SSR/df_A ; V_E = SSE/df_E
    
    anova_table = data.frame(
      요인 = c("A", "E", "계"),
      제곱합 = c(SSR, SSE, SST),
      자유도 = c(df_A, df_E, df_T),
      평균제곱합 = c(V_A, V_E, NA),
      F_0 = c(V_A/V_E, NA, NA),
      F_기각치 = c(qf(alpha, df_A, df_E, lower.tail = FALSE), NA, NA)
    )
    
    anova_table[,2:6] = round(anova_table[,2:6], 2)
    
    print(anova_table)
    
    return(list(X = X, alpha_hat = alpha_hat, anova_table = anova_table))
  }
}



#######################################################################################

# 이원배치법
twoway_anova <- function(y, A_num = 2, B_num = 2, R_num = 1, method = "null", alpha = 0.05){
  ## 이것도 마찬가지로 반복수가 같은 경우나 다른 경우나, 어차피 R코드상의 계산방식에는 변함이 없다.
  ## 따라서 반복수에 대한 구분을 굳이 할 필요가 없다 -> R_num은 반복횟수를 설정해주는 용도
  ## R_num이 1보다 큰 경우에 AB교호작용효과 열 만들기 / R_num = 1인 경우는 AB교호작용효과 스킵
  library(glue)
  n = length(y) ; one = c(rep(1, n))
  X = cbind(one)
  
  ## A 인자 열 만들기
  for (i in 1 : A_num){
    temp_A_vec = c(rep(0, n))
    for (a in ((i - 1) * (B_num * R_num) + 1) : (B_num * R_num * i)){
      temp_A_vec[a] = 1
    }
    X = cbind(X, temp_A_vec)
  }
  
  ## B 인자 열 만들기 : key point = i,j,k 구분하면서 계산하기
  for (j in 1 : B_num ){
    temp_B_vec = c()
    for (i in 1 : A_num){
      temp_b_vec = c(rep(0, (B_num * R_num)))
      for (k in ((j - 1) * R_num + 1) : (j * R_num)){
        temp_b_vec[k] = 1
      }
      temp_B_vec = c(temp_B_vec, temp_b_vec)
    }
    X = cbind(X, temp_B_vec)
  }

  colnames(X) = c("one", paste0("A_", 1:A_num), paste0("B_", 1:B_num))
  
  if (method == "one"){
    W = cbind(one)
    for (i in 1:(A_num - 1)){
      temp_W = X[,c(i + 1)] - X[,c(A_num + 1)]
      W = cbind(W, temp_W)
    }
    for (j in 1:(B_num - 1)){
      temp_W = X[,c(j + A_num + 1)] - X[,c(A_num + B_num + 1)]
      W = cbind(W, temp_W)
    }
    W = W[,-c(1)]
    
  }else if(method == "two"){
    X = X[,-c(1)]
    X = X[,-c(A_num, (A_num + B_num))]
  }else if(method == "null"){
    ## 분산분석표
    X_ = X[ , -c(1)]
    X_A = X_[ , c(1:A_num)] ; X_B = X_[ , c((A_num + 1) : (A_num + B_num))]   
    
    T_i.. = c(rep(0, A_num))
    for (i in 1:A_num){
      T_i..[i] = sum(X_A[,c(i)] * y)
    }
    
    T_.j. = c(rep(0, B_num))
    for (j in 1:B_num){
      T_.j.[j] = sum(X_B[,c(j)] * y)
    }  
    
    ybar_i.. = T_i.. / (B_num * R_num)
    ybar_.j. = T_.j. / (A_num * R_num)
    ybar = mean(y)
    
    mu_hat = ybar
    alpha_hat = ybar_i.. - ybar
    beta_hat = ybar_.j. - ybar
    
    S_A = sum(alpha_hat * T_i..) ; S_B = sum(beta_hat * T_.j.)
    SST = sum(y^2) - n * (ybar)^2 ; SSR = S_A + S_B ; SSE = SST - SSR
    
    df_A = (A_num - 1) ; df_B = (B_num - 1) ; df_E = ((A_num - 1) * (B_num - 1))
    
    V_A = S_A / df_A ; V_B = S_B / df_B ; V_E = SSE / df_E
    F_alpha_A = qf(alpha, df_A, df_E, lower.tail = FALSE)
    F_alpha_B = qf(alpha, df_B, df_E, lower.tail = FALSE)      
  }    
  
  ## 반복이 있는 경우
  if (R_num > 1){
    ## A X B 인자 열 만들기 : key point = 반복횟수 r만큼 점핑
    for (k in 1 : (A_num * B_num) ){
      temp_AB_vec = c(rep(0, n))
      for (t in ((k - 1) * R_num + 1) : (k * R_num)){
        temp_AB_vec[t] = 1
      }
      X = cbind(X, temp_AB_vec)
      colnames(X)[ncol(X)] <- paste0("AB_", k)
    }
  
    
    if (method == "one"){
      W = cbind(one)
      for (i in 1:(A_num - 1)){
        temp_W = X[,c(i + 1)] - X[,c(A_num + 1)]
        W = cbind(W, temp_W)
      }
      for (j in 1:(B_num - 1)){
        temp_W = X[,c(j + A_num + 1)] - X[,c(A_num + B_num + 1)]
        W = cbind(W, temp_W)
      }
      for (i in 1:(A_num - 1)){
        for (j in 1:(B_num - 1)){
          temp_W = (X[,c(i + 1)] - X[,c(A_num + 1)]) * (X[,c(j + A_num + 1)] - X[,c(A_num + B_num + 1)])
          W = cbind(W, temp_W)
        }
      }
      W = W[,-c(1)]
      
      mult_reg_res = mult_reg(W, y, alpha = alpha, coeff = TRUE)
      mu_hat = mult_reg_res$beta_hat[1]
      alpha_l_hat = -sum(mult_reg_res$beta_hat[2 : A_num])
      alpha_hat = c(mult_reg_res$beta_hat[2 : A_num], alpha_l_hat)
      beta_m_hat = -sum(mult_reg_res$beta_hat[(A_num + 1) : (A_num + B_num - 1)])
      beta_hat = c(mult_reg_res$beta_hat[(A_num + 1) : (A_num + B_num - 1)], beta_m_hat)
      
      alpha_beta_hat_vec = mult_reg_res$beta_hat[(A_num + B_num) : ((A_num - 1) * (B_num - 1) + (A_num + B_num) - 1)]
      alpha_beta_hat_mat = matrix(alpha_beta_hat_vec, ncol = (A_num - 1))
      alpha_beta_hat_mrow = colSums(alpha_beta_hat_mat)
      alpha_beta_hat_mat = rbind(alpha_beta_hat_mat, alpha_beta_hat_mrow)
      alpha_beta_hat_lcol = rowSums(alpha_beta_hat_mat)
      alpha_beta_hat_mat = cbind(alpha_beta_hat_mat, alpha_beta_hat_lcol)
      
      colnames(alpha_beta_hat) = c(paste0("A_", 1:A_num))
      rownames(alpha_beta_hat) = c(paste0("B_", 1:B_num))
      
      params = list(mu_hat = mu_hat, alpha_hat = alpha_hat, beta_hat = beta_hat, alpha_beta_hat = alpha_beta_hat)
      
      anova_table = mult_reg_res$anova_table
      
      return(list(X = W, params = params, anova_table = anova_table)) 
    }else if(method == "two"){
      X_AB = X[ , c( ((A_num - 1) * (B_num - 1)) : ((2 * A_num * B_num) - A_num - B_num) )]
      X = X[ , -c( ((A_num - 1) * (B_num - 1)) : ((2 * A_num * B_num) - A_num - B_num) )]
      X_AB = X_AB[ , -c(((A_num - 1)*B_num +1) : (A_num * B_num))]
      X_AB = X_AB[ , -c(seq(from = B_num, to = (B_num * (A_num - 1)), by = B_num))]
      
      X = cbind(X, X_AB)
      
      mult_reg_res = mult_reg(X, y, alpha = alpha, coeff = TRUE)
      gamma_0_hat = mult_reg_res$beta_hat[1]
      gamma_hat = mult_reg_res$beta_hat[2:(A_num)]
      tau_hat = mult_reg_res$beta_hat[(A_num + 1) : (A_num + B_num - 1)]
      gamma_tau_hat_vec = mult_reg_res$beta_hat[(A_num + B_num) : ((A_num - 1) * (B_num - 1) + (A_num + B_num) - 1)]
      gamma_tau_hat_mat = matrix(gamma_tau_hat_vec, ncol = (A_num - 1))
      
      params = list(gamma_0_hat = gamma_0_hat, gamma_hat = gamma_hat, tau_hat = tau_hat, gamma_tau_hat = gamma_tau_hat_mat)
      
      anova_table = mult_reg_res$anova_table
      
      return(list(X = X, params = params, anova_table = anova_table)) 
    }else if(method == "null"){
      X_ = X[ , -c(1)]
      X_AB = X_[ , c((A_num + B_num + 1) : (A_num + B_num + A_num * B_num) )]
      
      T_ij. = matrix(0, nrow = B_num, ncol = A_num)
      for (i in 1:(A_num)){
        for (j in 1:B_num){
          T_ij.[j,i] = sum( X_AB[,c(B_num * (i-1) + j)] * y)
        }
      }
      
      colnames(T_ij.) = c(paste0("A_", 1:A_num))
      rownames(T_ij.) = c(paste0("B_", 1:B_num))
      
      ybar_ij. = T_ij. / (R_num)
      
      alpha_beta_hat = matrix(0, nrow = B_num, ncol = A_num)
      for (i in 1:(A_num)){
        for (j in 1:B_num){
          alpha_beta_hat[j,i] = ybar_ij.[j,i] - ybar_i..[i] - ybar_.j.[j] + mu_hat
        }
      }
      colnames(alpha_beta_hat) = c(paste0("A_", 1:A_num))
      rownames(alpha_beta_hat) = c(paste0("B_", 1:B_num))
      
      params = list(mu_hat = mu_hat, alpha_hat = alpha_hat, beta_hat = beta_hat, alpha_beta_hat = alpha_beta_hat)
      
      S_AB = sum(alpha_beta_hat * T_ij.)
      SSR = SSR + S_AB ; SSE = SST - SSR
      
      df_AB = ((A_num - 1) * (B_num - 1)) ; df_E = ((A_num * B_num) * (R_num - 1))
      
      V_AB = S_AB / df_AB ; V_E = SSE / df_E
      F_alpha_A = qf(alpha, df_A, df_E, lower.tail = FALSE)
      F_alpha_B = qf(alpha, df_B, df_E, lower.tail = FALSE)
      F_alpha_AB = qf(alpha, df_AB, df_E, lower.tail = FALSE)
      
      anova_table = data.frame(
        요인 = c("A", "B", "AxB", "E", "계"),
        제곱합 = c(S_A, S_B, S_AB, SSE, SST),
        자유도 = c(df_A, df_B, df_AB, df_E, (A_num * B_num * R_num) - 1),
        평균제곱 = c(V_A, V_B, V_AB, V_E, NA),
        F_0 = c(V_A / V_E, V_B / V_E, V_AB / V_E, NA, NA),
        F_기각치 = c(F_alpha_A, F_alpha_B, F_alpha_AB, NA, NA)
      )
      
      anova_table[,c(2:6)] = round(anova_table[,c(2:6)], 2)
      print(anova_table)
      
      return(list(X = X, params = params, anova_table = anova_table))      
    }
  
  ## 반복이 없는 경우
  }else if (R_num == 1){
    if (method == "one"){
      mult_reg_res = mult_reg(W, y, alpha = alpha, coeff = TRUE)
      mu_hat = mult_reg_res$beta_hat[1]
      alpha_l_hat = -sum(mult_reg_res$beta_hat[2 : A_num])
      alpha_hat = c(mult_reg_res$beta_hat[2 : A_num], alpha_l_hat)
      beta_m_hat = -sum(mult_reg_res$beta_hat[(A_num + 1) : (A_num + B_num - 1)])
      beta_hat = c(mult_reg_res$beta_hat[(A_num + 1) : (A_num + B_num - 1)], beta_m_hat)
      
      params = list(mu_hat = mu_hat, alpha_hat = alpha_hat, beta_hat = beta_hat)
      
      anova_table = mult_reg_res$anova_table
      
      return(list(X = W, params = params, anova_table = anova_table)) 
    }else if(method == "two"){
      mult_reg_res = mult_reg(X, y, alpha = alpha, coeff = TRUE)
      gamma_0_hat = mult_reg_res$beta_hat[1]
      gamma_hat = mult_reg_res$beta_hat[2:(A_num)]
      tau_hat = mult_reg_res$beta_hat[(A_num + 1) : (A_num + B_num - 1)]
      
      params = list(gamma_0_hat = gamma_0_hat, gamma_hat = gamma_hat, tau_hat = tau_hat)
      
      anova_table = mult_reg_res$anova_table
      
      return(list(X = X, params = params, anova_table = anova_table)) 
    }else if (method == "null"){
      params = list(mu_hat = mu_hat, alpha_hat = alpha_hat, beta_hat = beta_hat)
      
      anova_table = data.frame(
        요인 = c("A", "B", "E", "계"),
        제곱합 = c(S_A, S_B, SSE, SST),
        자유도 = c(df_A, df_B, df_E, (A_num * B_num) - 1),
        평균제곱 = c(V_A, V_B, V_E, NA),
        F_0 = c(V_A / V_E, V_B / V_E, NA, NA),
        F_기각치 = c(F_alpha_A, F_alpha_B, NA, NA)
      )
      
      anova_table[,c(2:6)] = round(anova_table[,c(2:6)], 2)
      print(anova_table)
      
      return(list(X = X, params = params, anova_table = anova_table))      
    }
  }
}