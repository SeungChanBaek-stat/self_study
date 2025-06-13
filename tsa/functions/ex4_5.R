ex4_5_cal = function(dataset, type_cal = c("additive", "multiplicative")){
  library(forecast)
  library(glue)
  dataset_name <- deparse(substitute(build))
  
  if (type_cal == "additive"){
    
    m1 = decompose(dataset, type = type_cal)
    trend_add = trendcycle(m1)
    seasonal_add = seasonal(m1)
    irregular_add = remainder(m1)
    adjseasonal_add = dataset - seasonal_add
    pred_add = trend_add + seasonal_add
    
    pred_add_modified = pred_add
    pred_add_modified[1:6] = dataset[1:6]
    pred_add_modified[(n-5):n] = dataset[(n-5):n]
    residual_l = dataset - pred_add_modified
    MSE_modified = sum(residual_l^2) / n
    print(glue("MSE_modified = {MSE_modified}"))
    
    ts.plot(dataset, trend_add, ylab = dataset_name, lty = 1:2, col = c("blue", "red"),
            main = "원시계열과 추세•순환 성분(가법)")
    legend("topleft", lty = 1:2, col = c("blue", "red"), c("원시계열", "추세•순환(가법)"))
    
    ts.plot(dataset, seasonal_add, ylab = dataset_name, lty = 1:2, col = c("blue", "red"),
            main = "원시계열과 계절 성분(가법)")
    legend("topleft", lty = 1:2, col = c("blue", "red"), c("원시계열", "계절성분(가법)"))
    
    ts.plot(dataset, irregular_add, ylab = dataset_name, lty = 1:2, col = c("blue", "red"),
            main = "원시계열과 불규칙성분(가법)")
    legend("topleft", lty = 1:2, col = c("blue", "red"), c("원시계열", "불규칙성분(가법)"))
    
    ts.plot(dataset, adjseasonal_add, ylab = dataset_name, lty = 1:2, col = c("blue", "red"),
            main = "원시계열과 계절조정(가법)")
    legend("topleft", lty = 1:2, col = c("blue", "red"), c("원시계열", "계절조정(가법)"))
    
    ts.plot(dataset, pred_add, ylab = dataset_name, lty = 1:2, col = c("blue", "red"),
            main = "원시계열과 예측(가법)")
    legend("topleft", lty = 1:2, col = c("blue", "red"), c("원시계열", "예측(가법)"))
    
    ts.plot(seasonal_add, xlab = "year", ylab = "seasonal_add", lty = 3, col = "blue")
    ts.plot(irregular_add, xlab = "year", ylab = "irregular_add", lty = 3, col = "blue")
    
    acf(irregular_add, main = "불규칙성분의 ACF", na.action = na.pass)
  }else if(type_cal == "multiplicative"){
    m2 = decompose(dataset, type = c("multiplicative"))
    trend_mult = m2$trend
    seasonal_mult = m2$seasonal
    irregular_mult = m2$random
    adjseasonal_mult = dataset/seasonal_mult
    pred_mult = trend_mult * seasonal_mult
    
    pred_mult_modified = pred_mult
    pred_mult_modified[1:6] = dataset[1:6]
    pred_mult_modified[(n-5):n] = dataset[(n-5):n]
    residual_l = dataset - pred_mult_modified
    MSE_modified = sum(residual_l^2) / n
    print(glue("MSE_modified = {MSE_modified}"))
    
    ts.plot(dataset, trend_mult, ylab = dataset_name, lty = 1:2, col = c("blue", "red"),
            main = "원시계열과 추세•순환 성분(승법)")
    legend("topleft", lty = 1:2, col = c("blue", "red"), c("원시계열", "추세•순환(승법)"))
    
    ts.plot(dataset, seasonal_mult, ylab = dataset_name, lty = 1:2, col = c("blue", "red"),
            main = "원시계열과 계절 성분(승법)")
    legend("topleft", lty = 1:2, col = c("blue", "red"), c("원시계열", "계절성분(승법)"))
    
    ts.plot(dataset, irregular_mult, ylab = dataset_name, lty = 1:2, col = c("blue", "red"),
            main = "원시계열과 불규칙성분(승법)")
    legend("topleft", lty = 1:2, col = c("blue", "red"), c("원시계열", "불규칙성분(승법)"))
    
    ts.plot(dataset, adjseasonal_mult, ylab = dataset_name, lty = 1:2, col = c("blue", "red"),
            main = "원시계열과 계절조정(승법)")
    legend("topleft", lty = 1:2, col = c("blue", "red"), c("원시계열", "계절조정(승법)"))
    
    ts.plot(dataset, pred_mult, ylab = dataset_name, lty = 1:2, col = c("blue", "red"),
            main = "원시계열과 예측(승법)")
    legend("topleft", lty = 1:2, col = c("blue", "red"), c("원시계열", "예측(승법)"))
    
    ts.plot(seasonal_mult, xlab = "year", ylab = "seasonal_mult", lty = 3, col = "blue")
    ts.plot(irregular_mult, xlab = "year", ylab = "irregular_mult", lty = 3, col = "blue")
    
    acf(irregular_mult, main = "불규칙성분의 ACF", na.action = na.pass)
  }
  return(MSE_modified)
}