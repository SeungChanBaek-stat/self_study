#######################################################################################

# 이상점 탐지 함수
outliers = function(X, y, alpha = 0.05, MSE){
  library(glue)
  n = dim(X)[1] ; p = dim(X)[2] ; s = sqrt(MSE)
  one = c(rep(1,n)) ; X = cbind(one, X) ; In = diag(1, n)
  
  XtX_inv = solve(t(X) %*% X)
  H = X %*% XtX_inv %*% t(X)
  e = (In - H) %*% y
  
  h_ii = c(rep(0, n)) ; e_i = c(rep(0, n)) ; r_i = c(rep(0, n)) ; rstar_i = c(rep(0, n))
  DFFITS_i = c(rep(0, n)) ; D_i = c(rep(0, n)) ; M_i = c(rep(0, n)) ; AP_i = c(rep(0, n))
  COVRATIO_i = c(rep(0, n)) ; FVARATIO_i = c(rep(0, n))
  for (i in 1:n){
    x_i = X[i, ] ; h_ii[i] = t(x_i) %*% XtX_inv %*% x_i
    e_i[i] = e[i]
    r_i[i] = e_i[i] / (s * sqrt(1 - h_ii[i]))
    rstar_i[i] = r_i[i] * sqrt((n-p-2) / (n-p-1-(r_i[i])^2))
    DFFITS_i[i] = sqrt(h_ii[i] / (1 - h_ii[i])) * rstar_i[i]
    D_i[i] = h_ii[i] / ((p + 1) * (1 - h_ii[i])) * (r_i[i])^2
    M_i[i] = (n-1) * (h_ii[i] - 1/n)
    AP_i[i] = (1 - h_ii[i]) - ((e_i[i])^2 / ((n-p-1) * s^2))
    COVRATIO_i[i] = 1 / ((1 + ((rstar_i[i])^2 - 1) / (n-p-1))^(p+1) * (1 - h_ii[i]))
    FVARATIO_i[i] = (e_i[i])^2 / ((rstar_i[i])^2 * (1 - h_ii[i])^2 * s^2)
  }
  nums = c(seq(from = 1, to = n, by = 1))
  
  outlier_table <- data.frame(
    nums, e_i, h_ii, r_i, rstar_i,
    DFFITS_i, D_i, M_i, AP_i, COVRATIO_i, FVARATIO_i
  )
  
  colnames(outlier_table) <- c(
    "자료번호", "e_i", "h_ii", "r_i", "rstar_i",
    glue("DFFITS(i)"), glue("D(i)"), glue("M(i)"), glue("AP(i)"),
    glue("COVRATIO(i)"), glue("FVARATIO(i)")
  )
  
  outlier_table <- round(outlier_table, 4)
  
  print(outlier_table)
  
  hbar = mean(h_ii) ; t_alpha_n_p_2 = qt(alpha/2, n-p-2, lower.tail = FALSE)
  
  outlier_rule_table = data.frame(
    "(2hbar/ 3hbar)" = glue("({round(2* hbar, 4)} / {round(3 * hbar, 4)})"),
    "t_alpha_n_p_2" = glue("{round(t_alpha_n_p_2, 4)}")
  )
  print(outlier_rule_table)
  
  return(list(e_i = e_i, h_ii = h_ii, r_i = r_i, rstar_i = rstar_i))
}


