ex8_1 = function(sacf, spacf, n, z_bar, z_sd, diff = FALSE){
  library(glue)
  K = length(sacf) ; lag_seq = seq(1, K, 1)
  sacf_df = data.frame(sacf, lag_seq) ; spacf_df = data.frame(spacf, lag_seq)
  
  # 한 화면에 1행 2열로 배치
  op = par(mfrow=c(1,2), mar=c(4,4,2,1))
  
  # 1) Sample ACF (SACF)
  plot(lag_seq, sacf, type = "h", ylim = c(-1, 1), xlab = "Lag", ylab = "SACF", main = "Sample ACF")
  abline(h = 0) ; abline(h = c(-2/sqrt(n), 2/sqrt(n)), lty = 2, col = "blue")
  
  # 2) Sample PACF (SPACF)
  plot(lag_seq, spacf, type = "h", ylim = c(-1, 1), xlab = "Lag", ylab = "SPACF", main = "Sample PACF")
  abline(h = 0) ; abline(h = c(-2/sqrt(n), 2/sqrt(n)), lty = 2, col = "blue")
  
  # 원래 설정으로 복귀
  par(op)
  
  ## 단계 2
  
  ### AR 모형 판단
  AR_p_decision = c(rep(0, K))
  for (k in 1:K){
    if (abs(spacf[k]) > (1.96 / sqrt(n))){
      print(glue("|spacf[k]| = {spacf[k]}, crit = {1.96/sqrt(n)}"))
      print(glue("H_0 : phi_{k}{k} = 0 는 유의수준 alpha = 0.05에서 기각된다, AR({k}) is acceptable"))
      AR_p_decision[k] = 1
    }else{
      print(glue("H_0 : phi_{k}{k} = 0 는 유의수준 alpha = 0.05에서 기각되지 못한다, AR({k}) is unacceptable"))
    }
  }
  print(glue("\n"))
  
  ## MA 모형 판단
  MA_q_decision = c(rep(0, K))
  bartlett_sum = 1
  for (k in 1:K){
    var_rho_hat_k = bartlett_sum / n
    se_rho_hat_k = sqrt(var_rho_hat_k)
    if (abs(sacf[k]) > (1.96 * se_rho_hat_k)){
      print(glue("H_0 : rho_{k} = 0 는 유의수준 alpha = 0.05에서 기각된다, MA({k}) is acceptable"))
      MA_q_decision[k] = 1
    }else{
      print(glue("H_0 : rho_{k} = 0 는 유의수준 alpha = 0.05에서 기각되지 못한다, MA({k}) is unacceptable"))
    }
    bartlett_sum = bartlett_sum + 2*(sacf_vec[k])^2
  }
  
  ## 최종 모형 판단
  for (k in 1:(K-1)){
    AR_p_curr = AR_p_decision[k]
    if (AR_p_curr == 1){
      AR_p = k
    }else if(AR_p_curr == 0){
      AR_p = k-1
      break
    }
  }
  for (k in 1:(K-1)){
    MA_q_curr = MA_q_decision[k]
    if (MA_q_curr == 1){
      MA_q = k
    }else if(MA_q_curr == 0){
      MA_q = k-1
      break
    }
  }
  
  ## 변수의 절약
  if (AR_p < MA_q){
    print(glue("AR({AR_p})"))
    x = glue("AR({AR_p})")
    y = AR_p
  }else if(AR_p > MA_q){
    print(glue("MA({MA_q})"))
    x = glue("MA({MA_q})")
    y = MA_q
  }else if(AR_p == MA_q){
    print(glue("AR({AR_p}), MA({MA_q})"))
    x = glue("AR({AR_p}), MA({MA_q})")
    y = AR_p
  }
  print(glue("적절한 모형 : AR({AR_p}), MA({MA_q}) ; 변수의 절약에 의해 {x} 선택"))
  
  ## 상수항 포함여부 판단
  if (diff == FALSE){
    w_bar = z_bar
    gamma_0_hat = z_sd^2
    bartlett_sum = 1
    for (k in 1:y){
      # print(glue("sacf[k] = {sacf[k]}"))
      bartlett_sum = bartlett_sum + 2*sacf[k]
      # print(glue("bartlett_sum = {bartlett_sum}"))
    }
    s_wbar = sqrt(gamma_0_hat * bartlett_sum/n)
    print(glue("s_wbar = {s_wbar}"))
    t_0 = w_bar/s_wbar
    t_alpha = qt(p = 0.05, n - (y+1), lower.tail = FALSE)
    if (abs(t_0) >= t_alpha){
      print(glue("t_0 = {t_0}, t_alpha = {t_alpha}"))
      print(glue("H_0 : delta = 0 은 유의수준 alpha = 0.05 에서 기각한다, 상수항은 유의하다."))
    }else{
      print(glue("t_0 = {t_0}, t_alpha = {t_alpha}"))
      print(glue("H_0 : delta = 0 은 유의수준 alpha = 0.05 에서 기각하지 못한다, 상수항은 유의하지 않다."))
    }
  }else if (diff == TRUE){
    w_bar = z_bar
    s_wbar = z_sd
    t_0 = w_bar/s_wbar
    t_alpha = qt(p = 0.05, n - (y+1), lower.tail = FALSE)
    if (abs(t_0) >= t_alpha){
      print(glue("t_0 = {t_0}, t_alpha = {t_alpha}"))
      print(glue("H_0 : delta = 0 은 유의수준 alpha = 0.05 에서 기각한다, 상수항은 유의하다."))
    }else{
      print(glue("t_0 = {t_0}, t_alpha = {t_alpha}"))
      print(glue("H_0 : delta = 0 은 유의수준 alpha = 0.05 에서 기각하지 못한다, 상수항은 유의하지 않다."))
    }
  }
}