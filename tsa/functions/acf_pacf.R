## 자기공분산_k
gamma_hat_k = function(dataset, k){
  z_bar = mean(dataset) ; n = length(dataset)
  res = 0
  for (i in 1:(n-k)){
    z_t_z_bar = dataset[i] - z_bar
    z_tk_z_bar = dataset[i+k] - z_bar
    res = res + z_t_z_bar * z_tk_z_bar
  }
  res = res / n
  return(res)
}



## durbin_levinson 반복알고리즘 : SACF, SPACF 계산
durbin_levinson = function(dataset, K){
  library(glue)
  SACF = c(rep(0,K)) ; SPACF = c(rep(0, K)) ; rho_hat = c(rep(0, K))
  z_ts = ts(dataset)
  
  gamma_hat_0 = gamma_hat_k(z_ts, k = 0)
  for (i in 1:K){
    rho_hat[i] = gamma_hat_k(z_ts, k = i) / gamma_hat_0
    SACF[i] = rho_hat[i]
    print(glue("SACF_{i} = {rho_hat[i]}"))
  }
  
  phi_hat_mat = matrix(NA, nrow = K, ncol = K)
  
  for(i in 1:K){
    if(i == 1){
      phi_hat_mat[i,i] = rho_hat[i]
    } else {
      # 1) 대각성분 phi[i,i]
      num <- rho_hat[i] - sum(phi_hat_mat[i-1,1:(i-1)] * rho_hat[i - (1:(i-1))])
      denom <- 1 - sum(phi_hat_mat[i-1,1:(i-1)] * rho_hat[1:(i-1)])
      phi_hat_mat[i,i] <- num/denom
      
      # 2) 비대각성분 phi[i,1:(i-1)]
      phi_hat_mat[i,1:(i-1)] <- phi_hat_mat[i-1,1:(i-1)] - phi_hat_mat[i,i] * rev(phi_hat_mat[i-1,1:(i-1)])
    }
    SPACF[i] <- phi_hat_mat[i,i]
    print(glue("SPACF_{i} = {SPACF[i]}"))
  }
  # for (i in 1:K){
  #   for (j in 1:K){
  #     if ((i == 1) & (j == 1)){
  #       phi_hat_mat[i,j] = rho_hat[i]
  #       SPACF[i] = phi_hat_mat[i,j]
  #     }else if(i == j){
  #       l = 1 ; denum_sum = 0 ; denom_sum = 0
  #       while(l <= (i-1)){
  #         denum_sum = denum_sum + phi_hat_mat[(i-1), l] * rho_hat[i-l]
  #         denom_sum = denom_sum + phi_hat_mat[(i-1), l] * rho_hat[l]
  #         l = l + 1
  #       }
  #       denum = rho_hat[i] - denum_sum
  #       denom = 1 - denom_sum
  #       phi_hat_mat[i,j] = denum / denom
  #       print(phi_hat_mat)
  #     }else if(i > j){
  #       l = 1
  #       while(l <= (i-1)){
  #         phi_hat_mat[i,l] = phi_hat_mat[i-1,l] - phi_hat_mat[i,i] * phi_hat_mat[(i-1), i-l]
  #         l = l + 1
  #       }
  #       print(phi_hat_mat)
  #     }
  #   }
  #   SPACF[i] = phi_hat_mat[i,i]
  #   print(glue("SPACF_{i} = {SPACF[i]}"))
  # }
  return(list(SACF = SACF, SPACF = SPACF, phi_hat_mat = phi_hat_mat))
}