112/75
qt(0.05, df = 63, lower.tail = TRUE)
pnorm(-1.5,lower.tail = TRUE)
pnorm(-0.5, lower.tail = TRUE) - pnorm(-1.5, lower.tail = TRUE)
pnorm(0.5, lower.tail = TRUE) - pnorm(-0.5, lower.tail = TRUE)

1 - (pnorm(sqrt(5), lower.tail = TRUE) - pnorm(-sqrt(5), lower.tail = TRUE) )

1 - (pnorm(-sqrt(5) * (0.5), lower.tail = TRUE) - pnorm(-sqrt(5) * (2.5), lower.tail = TRUE) )

16*13 - 9 * 15

X = matrix(c(1,0,0,-1, 0,1,0,-1, 0,0,1,-1), nrow = 4); X

beta_hat = solve(t(X) %*% X) %*% t(X) ; beta_hat

c1 = c(1, 0 ,-1) ; C = rbind(c1) ; C

C %*% beta_hat

solve(t(X) %*% X)

C %*% solve(t(X) %*% X) %*% t(C)
diag(1, 4) - X %*% solve(t(X) %*% X) %*% t(X)


index_sol = c(1) ; index_given = c(0,2) ; index_test = NA

length(index_sol) ; length(index_given) ; length(index_test)

index_full <- sort(union(index_sol, index_given)) ; index_full

cbind(index_sol, index_given)

1.544

A = matrix(c(2*16.50, 2*17.20, 6.99, 6.99), nrow = 2) ; y = c(16.48, 3.38)

x = solve(A) %*% y ; x

A %*% x

x = solve(A, y) ; x




x1 = c(7, 1, 11, 11, 7, 11, 3, 1, 2, 21, 1, 11, 10)
x2 = c(26, 29, 56, 31, 52, 55, 71, 31, 54, 47, 40, 66, 68)
x3 = c(6, 15, 8, 8, 6, 9, 17, 22, 18, 4, 23, 9, 8)
x4 = c(60, 52, 20, 47, 33, 22, 6, 44, 22, 26, 34, 12, 12)
y = c(78.5, 74.3, 104.3, 87.6, 95.9, 109.2, 102.7, 72.5, 93.1, 115.9, 83.8, 113.3, 109.4)

X = cbind(x1, x2, x3, x4)



n = dim(X)[1] ; p = dim(X)[2]
one = c(rep(1, n))

colname_vec = colnames(X)

print(colname_vec)

max_num = 2^p
subset_k = c(rep(0, max_num))

library(sets)
# install.packages("sets")
a <- c("test1","test2","test3")
b <- set_power(a)
b
class(b)
B <- Vectorize(b)
B

total_subsets <- function(set) { 
  n <- length(set)
  masks <- 2^(1:n-1)
  res = sapply( 1:2^n-1, function(u) set[ bitwAnd(u, masks) != 0 ] )
  
  return(res)
}

# test = LETTERS[1:4]
# test_res = total_subsets(test)
# class(test_res[[2]])
# 
# test_res

colname_vec_res = total_subsets(colname_vec)

colname_vec_res

lapply(colname_vec_res,sort,decreasing=FALSE)

colname_vec_res
class(colname_vec_res[[2]])

class(colname_vec_res)
length(colname_vec_res)

colname_vec_res
