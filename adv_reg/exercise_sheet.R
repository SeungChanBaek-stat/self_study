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
