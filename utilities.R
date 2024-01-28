## generate samples from a GP
gp_sample <- function(x, a2 = 1, l = 1, sigma2 = 0.1){
  n <- nrow(x)
  mu <- rep(0, n)
  K <- matrix(nrow = n, ncol = n)
  for (i in 1:(n - 1)) {
    K[i, i] = a2 + sigma2
    for (j in (i + 1):n) {
      K[i, j] = a2 * exp(-sum((x[i,] - x[j,])^2) / l)
      K[j, i] = K[i, j]
    }
  }
  K[n, n] = a2 + sigma2
  return(t(mvtnorm::rmvnorm(1, sigma = K))) 
}

## debiased machine learning for partially linear model
dml_plm <- function(x, t, y, treg, yreg, nfold=5) {
  n <- nrow(x) # number of observations
  foldid <- rep.int(1:nfold, times = ceiling(n/nfold))[sample.int(n)] #define fold indices
  I <- split(1:n, foldid)  #split observation indices into folds  
  y_til <- t_til <- rep(NA, n)
  cat("fold: ")
  for(b in 1:length(I)){
    tfit <- treg(x[-I[[b]],], t[-I[[b]]]) #take a fold out
    yfit <- yreg(x[-I[[b]],], y[-I[[b]]]) # take a fold out
    t_hat <- predict(tfit, x[I[[b]],], type="response") #predict the left-out fold 
    y_hat <- predict(yfit, x[I[[b]],], type="response") #predict the left-out fold 
    t_til[I[[b]]] <- (t[I[[b]]] - t_hat) #record residual for the left-out fold
    y_til[I[[b]]] <- (y[I[[b]]] - y_hat) #record residial for the left-out fold
    cat(b," ")
  }
  rfit <- lm(y_til ~ t_til)
  rfitSummary<- summary(rfit)
  coef.est <-  rfitSummary$coef[2] #extract coefficient
  se <- rfitSummary$coef[2,2]  #record robust standard error
  cat(sprintf("\ncoef (se) = %g (%g)\n", coef.est , se))  #printing output
  return(list(coef.est=coef.est , se=se, t_til=t_til, y_til=y_til)) #save output and residuals 
}

## partialling out covariates
get_residuals <- function(x, y, model, nfold=5) {
  n <- nrow(x) # number of observations
  foldid <- rep.int(1:nfold, times = ceiling(n/nfold))[sample.int(n)] #define fold indices
  I <- split(1:n, foldid)  #split observation indices into folds  
  y_res <- rep(NA, n)
  cat("fold: ")
  for(b in 1:length(I)){
    yfit <- model(x[-I[[b]],], y[-I[[b]]]) # take a fold out
    y_hat <- predict(yfit, x[I[[b]],], type="response") #predict the left-out fold 
    y_res[I[[b]]] <- (y[I[[b]]] - y_hat) #record residial for the left-out fold
    cat(b," ")
  }
  return(y_res) #save residuals 
}

## format percentages
format.perc <- function(probs, digits)
{
  paste(format(100 * probs,
               trim = TRUE,
               scientific = FALSE,
               digits = digits),
        "%")
}