# library(future.apply)

## check if matrix is psd and make it psd
is_positive_definite <- function(mat) {
  return(all(eigen(mat, only.values = TRUE)$values > 0))
}

make_positive_definite <- function(mat, increment = 1e-6, max_attempts = 100) {
  attempts <- 0
  new_mat <- mat
  
  while (!is_positive_definite(new_mat) && attempts < max_attempts) {
    diag(new_mat) <- diag(new_mat) + increment
    attempts <- attempts + 1
  }
  
  if (attempts == max_attempts) {
    warning("Matrix could not be made positive definite within the maximum attempts.")
  }
  
  return(new_mat)
}

## generate samples from a GP with RBF kernel and nugget effect
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

## generate a covariance matrix with RBF kernel
rbf_cov <- function(s, a2 = 1, l = 1){
  n <- nrow(s)
  mu <- rep(0, n)
  K <- matrix(nrow = n, ncol = n)
  for (i in 1:(n - 1)) {
    K[i, i] = a2 
    for (j in (i + 1):n) {
      K[i, j] = a2 * exp(-sum((s[i,] - s[j,])^2) / l)
      K[j, i] = K[i, j]
    }
  }
  K[n, n] = a2
  return(K + diag(1e-6, n)) 
}

## Find inverse square root of matrix X
inv_sqrt <- function(X){
  e <- eigen(X)
  v <- e$vectors
  return(v %*% diag(1/sqrt(e$values)) %*% t(v))
}

dml_plr_gam <- function(df, y, d, x_spline, x_other=NULL,
                        K = 5, k_spline = 200, seed = 1) {
  
  # -------- 1. preparation --------
  is_dt <- inherits(df, "data.table")
  if (!is_dt) df <- as.data.frame(df)  # stay a data.frame if user prefers
  
  N      <- nrow(df)
  set.seed(seed)
  folds  <- sample(rep(1:K, length.out = N), N)
  
  resY <- resD <- numeric(N)
  
  spline_part <- sprintf("s(%s, k = %d)",
                         paste(x_spline, collapse = ", "), k_spline)
  
  # Construct formula strings
  if (is.null(x_other)) {
    f_Y <- as.formula(sprintf("%s ~ %s", y, spline_part))
    f_D <- as.formula(sprintf("%s ~ %s", d, spline_part))
  } else {
    x_part <- paste(x_other, collapse = " + ")
    f_Y <- as.formula(sprintf("%s ~ %s + %s", y, spline_part, x_part))
    f_D <- as.formula(sprintf("%s ~ %s + %s", d, spline_part, x_part))
  }
  
  # -------- 2. K‑fold cross‑fitting --------
  for (k in 1:K) {
    idx_train <- which(folds != k)
    idx_test  <- which(folds == k)
    
    fit_Y <- mgcv::gam(f_Y, data = df[idx_train, , drop = FALSE])
    fit_D <- mgcv::gam(f_D, data = df[idx_train, , drop = FALSE])
    
    resY[idx_test] <- df[[y]][idx_test] - predict(fit_Y, newdata = df[idx_test, , drop = FALSE])
    resD[idx_test] <- df[[d]][idx_test] - predict(fit_D, newdata = df[idx_test, , drop = FALSE])
  }
  
  # -------- 3. residual‑on‑residual regression --------
  reg      <- lm(resY ~ resD)
  vcov_hc  <- sandwich::vcovHC(reg, type = "HC3")
  
  theta    <- coef(reg)["resD"]
  se       <- sqrt(vcov_hc["resD", "resD"])
  ci_95    <- theta + c(-1, 1) * qnorm(0.975) * se   # 95% CI
  
  list(
    coef      = theta,
    se        = se,
    ci_95     = ci_95,
    resD = resD,
    resY = resY,
    summary   = lmtest::coeftest(reg, vcov. = vcov_hc)
  )
}

## debiased machine learning for partially linear model
# dml_plm <- function(x, t, y, treg, yreg, nfold=5) {
#   use_X   <- !is.null(x) && ncol(as.matrix(x)) > 0
#   n       <- if (use_X) nrow(x) else length(t)
#   
#   foldid <- rep.int(1:nfold, times = ceiling(n/nfold))[sample.int(n)] #define fold indices
#   I <- split(1:n, foldid)  #split observation indices into folds
#   y_til <- t_til <- rep(NA, n)
#   cat("fold: ")
#   for(b in 1:length(I)){
#     tfit <- treg(x[-I[[b]],], t[-I[[b]]]) #take a fold out
#     yfit <- yreg(x[-I[[b]],], y[-I[[b]]]) # take a fold out
#     t_hat <- predict(tfit, x[I[[b]],], type="response") #predict the left-out fold
#     y_hat <- predict(yfit, x[I[[b]],], type="response") #predict the left-out fold
#     t_til[I[[b]]] <- (t[I[[b]]] - t_hat) #record residual for the left-out fold
#     y_til[I[[b]]] <- (y[I[[b]]] - y_hat) #record residial for the left-out fold
#     cat(b," ")
#   }
#   rfit <- lm(y_til ~ t_til)
#   rfitSummary<- summary(rfit)
#   coef.est <-  rfitSummary$coef[2] #extract coefficient
#   se <- rfitSummary$coef[2,2]  #record robust standard error
#   cat(sprintf("\ncoef (se) = %g (%g)\n", coef.est , se))  #printing output
#   return(list(coef.est=coef.est , se=se, t_til=t_til, y_til=y_til)) #save output and residuals
# }

dml_plm <- function(x        = NULL,          # covariate matrix / data.frame OR NULL
                    d, y,                    # treatment (d) & outcome (y) vectors
                    dreg = NULL, yreg = NULL,  # learners (ignored if x is NULL)
                    nfold = 5) {
  
  # ---------------------------- preparation ----------------------------
  use_X   <- !is.null(x) && ncol(as.matrix(x)) > 0
  n       <- if (use_X) nrow(x) else length(d)
  
  if (length(d) != n || length(y) != n)
    stop("Lengths of d and y must match nrow(x).")
  
  foldid  <- rep_len(1:nfold, n)[sample.int(n)]
  folds   <- split(seq_len(n), foldid)
  
  y_til <- d_til <- numeric(n)
  
  cat("fold: ")
  # ---------------------------- cross–fitting --------------------------
  for (b in seq_along(folds)) {
    idx_out <- folds[[b]]           # validation indices
    idx_in  <- setdiff(seq_len(n), idx_out)
    
    if (use_X) {
      # fit ML learners on training fold
      dfit <- dreg(x[idx_in, , drop = FALSE], d[idx_in])
      yfit <- yreg(x[idx_in, , drop = FALSE], y[idx_in])
      
      d_hat <- predict(dfit, x[idx_out, , drop = FALSE], type = "response")
      y_hat <- predict(yfit, x[idx_out, , drop = FALSE], type = "response")
    } else {
      # no covariates → use out-of-fold means
      d_hat <- rep(mean(d[idx_in]), length(idx_out))
      y_hat <- rep(mean(y[idx_in]), length(idx_out))
    }
    
    d_til[idx_out] <- d[idx_out] - d_hat
    y_til[idx_out] <- y[idx_out] - y_hat
    cat(b, " ")
  }
  
  # ---------------------------- second stage ---------------------------
  rfit <- lm(y_til ~ d_til)
  rsum <- summary(rfit)
  
  coef.est <- rsum$coef[2, 1]
  se       <- rsum$coef[2, 2]
  
  cat(sprintf("\ncoef (se) = %.4g (%.4g)\n", coef.est, se))
  invisible(list(coef.est = coef.est,
                 se       = se,
                 resD    = d_til,
                 resY    = y_til))
}

dml_plm_lm <- function(x        = NULL,         # covariate matrix / data.frame  OR  NULL
                       d, y,                   # treatment & outcome vectors (length n)
                       nfold   = 5) {
  
  ## ------------- preparation ------------------------------------------------
  use_X <- !is.null(x) && ncol(as.matrix(x)) > 0
  n     <- if (use_X) nrow(x) else length(d)
  
  if (length(d) != n || length(y) != n)
    stop("Lengths of d and y must match nrow(x).")
  
  foldid <- rep_len(seq_len(nfold), n)[sample.int(n)]   # random folds
  folds  <- split(seq_len(n), foldid)
  
  y_til <- d_til <- numeric(n)          # cross-fitted residuals
  
  cat("fold: ")
  ## ------------- cross-fitting ---------------------------------------------
  for (b in seq_along(folds)) {
    idx_out <- folds[[b]]              # validation indices
    idx_in  <- setdiff(seq_len(n), idx_out)
    
    if (use_X) {
      
      ## --------- 1. regress d on X (training fold) --------------------------
      dfit <- lm(d[idx_in] ~ ., data = as.data.frame(x[idx_in, , drop = FALSE]))
      
      ## --------- 2. regress y on X (training fold) --------------------------
      yfit <- lm(y[idx_in] ~ ., data = as.data.frame(x[idx_in, , drop = FALSE]))
      
      ## --------- 3. predict on held-out fold --------------------------------
      d_hat <- predict(dfit, newdata = as.data.frame(x[idx_out, , drop = FALSE]))
      y_hat <- predict(yfit, newdata = as.data.frame(x[idx_out, , drop = FALSE]))
      
    } else {
      ## no covariates – use training-fold means
      d_hat <- rep(mean(d[idx_in]), length(idx_out))
      y_hat <- rep(mean(y[idx_in]), length(idx_out))
    }
    
    ## --------- store residuals ---------------------------------------------
    d_til[idx_out] <- d[idx_out] - d_hat
    y_til[idx_out] <- y[idx_out] - y_hat
    cat(b, " ")
  }
  
  ## ------------- second-stage regression -----------------------------------
  rfit <- lm(y_til ~ d_til)
  rsum <- summary(rfit)
  
  coef.est <- rsum$coef[2, 1]
  se.est   <- rsum$coef[2, 2]
  
  cat(sprintf("\ncoef (se) = %.4g (%.4g)\n", coef.est, se.est))
  
  invisible(list(coef.est = coef.est,
                 se       = se.est,
                 resD     = d_til,
                 resY     = y_til))
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
format.perc <- function(probs, digits) {
  paste(format(100 * probs,
               trim = TRUE,
               scientific = FALSE,
               digits = digits),
        "%")
}

## Build splines
make_knots <- function(z_all, L, degree = 3, boundary_expand = 0.01) {
  n_int <- max(L - (degree + 1), 0)
  probs <- if (n_int > 0) (1:n_int) / (n_int + 1) else numeric(0)
  internal <- if (n_int > 0) as.numeric(quantile(z_all, probs)) else numeric(0)
  rng <- range(z_all, finite = TRUE)
  pad <- diff(rng) * boundary_expand
  list(internal = internal,
       Boundary.knots = c(rng[1] - pad, rng[2] + pad),
       degree = degree)
}

build_bases <- function(d_mat, L, degree = 3) {
  d_mean <- mean(d_mat); d_sd <- sd(as.numeric(d_mat))
  z_mat  <- (d_mat - d_mean) / d_sd
  z_all  <- as.numeric(z_mat)
  
  knot_info <- make_knots(z_all, L, degree)
  B_pool <- bSpline(
    z_all, knots = knot_info$internal, degree = degree,
    Boundary.knots = knot_info$Boundary.knots, intercept = TRUE
  )
  col_means <- colMeans(B_pool)
  
  N <- nrow(d_mat); K <- ncol(d_mat)
  B_list <- vector("list", K)
  for (k in 1:K) {
    Bk_unc <- bSpline(
      z_mat[, k],
      knots = knot_info$internal, degree = degree,
      Boundary.knots = knot_info$Boundary.knots, intercept = TRUE
    )
    B_list[[k]] <- sweep(Bk_unc, 2, col_means, "-")
  }
  
  list(
    z_mat = z_mat,
    d_mean = d_mean, d_sd = d_sd,
    B_list = B_list,
    knot_info = knot_info,
    col_means = col_means
  )
}

# ---------- Recover treatment effects ----------
# ----- Pointwise causal component on original d-scale -----
# f(d) = beta_lin * z + theta' B_centered(z)
f_causal <- function(d_vec, beta_lin, theta, aux) {
  z <- (d_vec - aux$d_mean) / aux$d_sd
  # uncentered basis
  B_raw <- bSpline(
    z,
    knots = aux$knot_info$internal,
    degree = aux$knot_info$degree,
    Boundary.knots = aux$knot_info$Boundary.knots,
    intercept = TRUE
  )
  # center columns same as training
  Bc <- sweep(B_raw, 2, aux$col_means, "-")
  L_basis <- ncol(Bc)
  if (length(theta) < L_basis) stop("theta length < basis columns: ",
                                    length(theta), " vs ", L_basis)
  if (length(theta) > L_basis) theta <- theta[seq_len(L_basis)]
  as.numeric(beta_lin * z + as.vector(Bc %*% as.numeric(theta)))
}

# ----- Pointwise marginal derivative (ACD integrand) on original d-scale -----
# TE(d) = (beta_lin + theta' B'(z)) / sd
te_point <- function(d_vec, beta_lin, theta, aux) {
  z <- (d_vec - aux$d_mean) / aux$d_sd
  Bprime <- dbs(
    z,
    knots = aux$knot_info$internal,
    degree = aux$knot_info$degree,
    Boundary.knots = aux$knot_info$Boundary.knots,
    intercept = TRUE
  )
  L_basis <- ncol(Bprime)
  if (length(theta) < L_basis) stop("theta length < basis columns: ",
                                    length(theta), " vs ", L_basis)
  if (length(theta) > L_basis) theta <- theta[seq_len(L_basis)]
  as.numeric(beta_lin + as.vector(Bprime %*% as.numeric(theta))) / aux$d_sd
}

# ----- Finite-change effect for a shift Delta -----
# ΔY(d;Δ) = f(d+Δ) - f(d)
# Efficiently exploits that z1 - z0 = Δ / sd and uses centered bases.
delta_effect_vec <- function(d_vec, Delta, beta_lin, theta, aux) {
  z0 <- (d_vec - aux$d_mean) / aux$d_sd
  z1 <- z0 + Delta / aux$d_sd
  
  B0_raw <- bSpline(
    z0,
    knots = aux$knot_info$internal,
    degree = aux$knot_info$degree,
    Boundary.knots = aux$knot_info$Boundary.knots,
    intercept = TRUE
  )
  B1_raw <- bSpline(
    z1,
    knots = aux$knot_info$internal,
    degree = aux$knot_info$degree,
    Boundary.knots = aux$knot_info$Boundary.knots,
    intercept = TRUE
  )
  
  B0c <- sweep(B0_raw, 2, aux$col_means, "-")
  B1c <- sweep(B1_raw, 2, aux$col_means, "-")
  L_basis <- ncol(B0c)
  if (length(theta) < L_basis) stop("theta length < basis columns: ",
                                    length(theta), " vs ", L_basis)
  if (length(theta) > L_basis) theta <- theta[seq_len(L_basis)]
  
  lin <- beta_lin * (Delta / aux$d_sd)
  nl  <- as.numeric((B1c - B0c) %*% as.numeric(theta))
  lin + nl
}

# ===== High-level wrappers =====

# Average causal derivative over the empirical D
average_causal_derivative <- function(d_mat, beta_lin, theta, aux) {
  mean(te_point(as.numeric(d_mat), beta_lin, theta, aux))
}

# Average finite change for shift Delta
average_shift_effect <- function(d_mat, Delta, beta_lin, theta, aux) {
  mean(delta_effect_vec(as.numeric(d_mat), Delta, beta_lin, theta, aux))
}
