## 단순지수평활법
simple_exp_smt = function(dataset, w, init_val, l){
  S_0 = init_val ; n = length(dataset) ; Z = dataset
  Z_hat_l_vec = c(rep(0, n))
  
  for (i in 1:n){
    if (i == 1){
      Z_hat_l_vec[i] = S_0
    }else{
      Z_hat_l_vec[i] = w * Z[i-1] + (1- w) * Z_hat_l_vec[i-1]
    }
  }
  residual_l = as.vector(Z) - Z_hat_l_vec
  MSE = sum(residual_l^2)/n
  MAE = sum(abs(residual_l))/n
  MAPE = 100 * sum(abs(residual_l/Z))/n
  
  return(list(Z_hat_l = Z_hat_l_vec, residual_l = residual_l, MSE = MSE, MAE = MAE, MAPE = MAPE))
}



## 이중지수평활법
double_exp_smt = function(dataset, w, l){
  n = length(dataset) ; Z = dataset ; t = seq(1, n, 1)
  lm_fit = lm(dataset ~ t)
  beta_0_init = lm_fit$coefficients[1]
  beta_1_init = lm_fit$coefficients[2]
  
  S_1_init = beta_0_init - (((1-w)/w) * beta_1_init)
  S_2_init = beta_0_init - (2*((1-w)/w) * beta_1_init)
  
  S_1_n_vec = c(rep(0, n)) ; S_2_n_vec = c(rep(0, n))
  
  Z_hat_l_vec = c(rep(0, n))
  
  for (i in 1:n){
    if (i == 1){
      S_1_n_vec[i] = w * Z[i] + (1-w) * S_1_init
      S_2_n_vec[i] = w * S_1_n_vec[i] + (1-w) * S_2_init
      # Z_hat_l_vec[i] = Z[i]
    }else{
      S_1_n_vec[i] = w * Z[i] + (1-w) * S_1_n_vec[i-1]
      S_2_n_vec[i] = w * S_1_n_vec[i] + (1-w) * S_2_n_vec[i-1]
      # Z_hat_l_vec[i] = ((2 + (w/(1-w))*l) * S_1_n_vec[i]) - ((1 + (w/(1-w))*l) * S_2_n_vec[i])
    }
    Z_hat_l_vec[i] = ((2 + (w/(1-w))*l) * S_1_n_vec[i]) - ((1 + (w/(1-w))*l) * S_2_n_vec[i])
  }
  
  residual_l = as.vector(Z) - Z_hat_l_vec
  MSE = sum(residual_l^2)/n
  MAE = sum(abs(residual_l))/n
  MAPE = 100 * sum(abs(residual_l/Z))/n
  
  return(list(Z_hat_l = Z_hat_l_vec, residual_l = residual_l, MSE = MSE, MAE = MAE, MAPE = MAPE))
}



## 삼중지수평활법
triple_exp_smt = function(dataset, w, l){
  n = length(dataset) ; Z = dataset ; t = seq(1, n, 1)
  lm_fit = lm(dataset ~ t + I(t^2/2))
  beta_0_init = lm_fit$coefficients[1]
  beta_1_init = lm_fit$coefficients[2]
  beta_2_init = lm_fit$coefficients[3]
  
  S_1_init = beta_0_init - (((1-w)/w) * beta_1_init) + ((1-w)*(2-w)/(2*(w^2)))*beta_2_init
  S_2_init = beta_0_init - (2*((1-w)/w) * beta_1_init) + (2*(1-w)*(3-2*w)/(2*(w^2)))*beta_2_init
  S_3_init = beta_0_init - (3*((1-w)/w) * beta_1_init) + (3*(1-w)*(4-3*w)/(2*(w^2)))*beta_2_init
  
  S_1_n_vec = c(rep(0, n)) ; S_2_n_vec = c(rep(0, n)) ; S_3_n_vec = c(rep(0, n))
  
  Z_hat_l_vec = c(rep(0, n))
  
  for (i in 1:n){
    if (i == 1){
      S_1_n_vec[i] = w * Z[i] + (1-w) * S_1_init
      S_2_n_vec[i] = w * S_1_n_vec[i] + (1-w) * S_2_init
      S_3_n_vec[i] = w * S_2_n_vec[i] + (1-w) * S_3_init
      # Z_hat_l_vec[i] = Z[i]
    }else{
      S_1_n_vec[i] = w * Z[i] + (1-w) * S_1_n_vec[i-1]
      S_2_n_vec[i] = w * S_1_n_vec[i] + (1-w) * S_2_n_vec[i-1]
      S_3_n_vec[i] = w * S_2_n_vec[i] + (1-w) * S_3_n_vec[i-1]
      # Z_hat_l_vec[i] = ((2 + (w/(1-w))*l) * S_1_n_vec[i]) - ((1 + (w/(1-w))*l) * S_2_n_vec[i])
    }
    S_1_n_i = (3 + ((w*(6-5*w)/(2*(1-w)^2))*l + (w^2/(2*(1-w)^2))*l^2) )* S_1_n_vec[i]
    S_2_n_i = (3 + ((w*(5-4*w)/((1-w)^2))*l + (w^2/(1-w)^2)*l^2) )* S_2_n_vec[i]
    S_3_n_i = (1 + ((w*(4-3*w)/(2*(1-w)^2))*l + (w^2/(2*(1-w)^2))*l^2) )* S_3_n_vec[i]
    Z_hat_l_vec[i] =  S_1_n_i - S_2_n_i + S_3_n_i
  }
  
  residual_l = as.vector(Z) - Z_hat_l_vec
  MSE = sum(residual_l^2)/n
  MAE = sum(abs(residual_l))/n
  MAPE = 100 * sum(abs(residual_l/Z))/n
  
  return(list(Z_hat_l = Z_hat_l_vec, residual_l = residual_l, MSE = MSE, MAE = MAE, MAPE = MAPE))
}



## w 가중치 선택함수 (SSE를 최소로 하는 w)
exp_weight_sel = function(w_vec, dataset, function_name, l){
  library(glue)
  m = length(w_vec) ; MSE_vec = c(rep(0, m))
  if (function_name == "simple_exp_smt"){
    for (i in 1:m){
      simp_exp_temp = simple_exp_smt(dataset, w_vec[i], mean(dataset), l)
      MSE_vec[i] = simp_exp_temp$MSE
    }
  }else if(function_name == "double_exp_smt"){
    for (i in 1:m){
      doub_exp_temp = double_exp_smt(dataset, w_vec[i], l)
      MSE_vec[i] = doub_exp_temp$MSE
    }
  }else if(function_name == "triple_exp_smt"){
    for (i in 1:m){
      tri_exp_temp = triple_exp_smt(dataset, w_vec[i], l)
      MSE_vec[i] = tri_exp_temp$MSE
    }
  }
  plot(w_vec, MSE_vec, xlab = "weight", ylab = "MSE", type = "o",
       main = "1-시차 후 예측오차의 제곱합")
  print(glue("{function_name} 의 최적 w : {w_vec[which.min(MSE_vec)]}, MSE = min(MSE)"))
  return(w_opt = w_vec[which.min(MSE_vec)])
}