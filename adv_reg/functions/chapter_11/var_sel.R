
library(glue)
library(olsrr)

#######################################################################################

# 전진선택법

forward.sel <- function(data, alpha, method = c("adj.R", "Mallow") ){
  
  p = ncol(data)
  b.X = data[,-ncol(data)]
  b.data = data.frame(Salary = data[,ncol(data)])
  b.val.names = colnames(b.X)
  b.lm.fit = lm(Salary ~ 1, data = b.data)
  
  lm.fit.full = lm(Salary ~ ., data = data)
  
  if (method == "adj.R"){
    b.cri = - 100 # sufficiently small
  }else if(method == "Mallow"){
    b.cri = 1e+5 # sufficiently large
  }
  
  total.sel_val = NULL
  total.cri = NULL
  i = 1
  repeat{
    if (method == "adj.R"){
      # calculate adj.R
      T.adj = numeric(length = ncol(b.X))
      T.adata = vector(mode = "list", length = ncol(b.X))
      for (j in 1:ncol(b.X)){
        x = data.frame(b.X[,j])
        colnames(x) = b.val.names[j]
        T.adata[[j]] = data.frame(x, b.data)
        T.adj[j] = summary(lm(Salary ~ ., data = T.adata[[j]]))$adj
      }
      # candidate
      cri = max(T.adj)
      loc_cri = which.max(T.adj)
      adata = T.adata[[loc_cri]]
      # stopping rule
      anova1 = anova(b.lm.fit, lm(Salary ~ ., data = adata))
      numerator = anova1$`Sum of Sq`[2]
      anova2 = anova(b.lm.fit, lm.fit.full)
      denominator = anova2$RSS[2] / anova2$Res.Df[2]
      F_0 = numerator / denominator
      rej_cri = qf(alpha, 1, anova2$Res.Df[2], lower.tail = FALSE)
      stop = b.cri > cri | F_0 < rej_cri
    }else if(method == "Mallow"){
      # calculate Mallow's Cp
      T.Mallow = numeric(length = ncol(b.X))
      T.adata = vector(mode = "list", length = ncol(b.X))
      for (j in 1:ncol(b.X)){
        x = data.frame(b.X[,j])
        colnames(x) = b.val.names[j]
        T.adata[[j]] = data.frame(x, b.data)
        T.Mallow[j] = ols_mallows_cp(lm(Salary ~ ., data = T.adata[[j]]),
                                     lm(Salary ~ ., data = data))
        
      }
      # candidate
      cri = min(T.Mallow)
      loc_cri = which.min(T.Mallow)
      adata = T.adata[[loc_cri]]
      # stopping rule
      anova1 = anova(b.lm.fit, lm(Salary ~ ., data = adata))
      numerator = anova1$`Sum of Sq`[2]
      anova2 = anova(b.lm.fit, lm.fit.full)
      denominator = anova2$RSS[2] / anova2$Res.Df[2]
      F_0 = numerator / denominator
      rej_cri = qf(alpha, 1, anova2$Res.Df[2], lower.tail = FALSE)
      stop = b.cri < cri | F_0 < rej_cri
    }
    if (stop){
      break
    }else{
      selected.val = b.val.names[loc_cri]
      total.sel_val = rbind(total.sel_val, selected.val)
      total.cri = rbind(total.cri, cri)
      b.data = adata
      b.X = b.X[, -loc_cri]
      b.val.names = b.val.names[-loc_cri]
      b.cri = cri
      b.lm.fit = lm(Salary ~ ., data = b.data)
    }
  }
  colnames(total.sel_val) = "selected.values"
  rownames(total.sel_val) = seq_along(total.sel_val)
  if (method == "adj.R"){
    colnames(total.cri) = "adj.R"
  }else if(method == "Mallow"){
    colnames(total.cri) = "Mallow.Cp"
  }
  rownames(total.cri) = seq_along(total.cri)
  return(list(selected.values = total.sel_val, cri = total.cri, final.data = b.data))  
}


#######################################################################################

# 후진제거법

back.eli = function(data, alpha, method = c("adj.R", "Mallow")){
  p = ncol(data)
  lm.fit.full = lm(Salary ~ ., data = data)
  
  if (method == "adj.R"){
    b.cri = summary(lm.fit.full)$adj
  }else if(method == "Mallow"){
    b.cri = p # k + 1
  }
  
  total.eli_val = NULL
  total.cri = NULL
  i = 1
  repeat{
    if (method == "adj.R"){
      # calculate adj.R
      T.adj = numeric(ncol(data) - 1)
      for (j in 1:(ncol(data) - 1)){
        adata = data[, -j]
        T.adj[j] = summary(lm(Salary ~ ., data = adata))$adj
      }
      # candidate
      cri = max(T.adj)
      loc_cri = which.max(T.adj)
      # stopping rule
      stop = b.cri > cri | summary(lm(Salary ~ ., data = data))$coef[-1, 4][loc_cri] < alpha
    }else if(method == "Mallow"){
      # calculate Mallow's Cp
      T.Mallow = numeric(ncol(data) - 1)
      for (j in 1:(ncol(data) - 1)){
        adata = data[, -j]
        T.Mallow[j] = ols_mallows_cp(lm(Salary ~ ., data = adata), lm.fit.full)
      }
      # candidate
      cri = min(T.Mallow)
      loc_cri = which.min(T.Mallow)
      # stopping rule
      stop = b.cri < cri | summary(lm(Salary ~ ., data = data))$coef[-1, 4][loc_cri] < alpha
    }
    if (stop){
      break
    }else{
      eliminated.val = colnames(data)[loc_cri]
      total.eli_val = rbind(total.eli_val, eliminated.val)
      total.cri = rbind(total.cri, cri)
      data = data[, -loc_cri]
      b.cri = cri
    }
    i = i + 1
    if (i >= p){
      break
    }
  }
  colnames(total.eli_val) = "eliminated.values"
  rownames(total.eli_val) = seq_along(total.eli_val)
  if (method == "adj.R"){
    colnames(total.cri) = "adj.R"
  }else if(method == "Mallow"){
    colnames(total.cri) = "Mallow.Cp"
  }
  rownames(total.cri) = seq_along(total.cri)
  return(list(eliminated.values = total.eli_val, cri = total.cri, final.data = data))
}

