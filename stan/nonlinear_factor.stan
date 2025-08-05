// Y,D,X all vary across unit and time, but the direct dependence is invariant.
functions {
  matrix sqrtm(matrix A) {
    int m = rows(A);
    vector[m] root_root_evals = sqrt(sqrt(eigenvalues_sym(A)));
    matrix[m, m] evecs = eigenvectors_sym(A);
    matrix[m, m] eprod = diag_post_multiply(evecs, root_root_evals);
    return tcrossprod(eprod);
  }
}
data {
  int<lower=1> K;                 // # treatments and outcomes (units or time points)
  int<lower=1> N;                 // # observations
  int<lower=1> M;                 // # latent confounders
  array[N] vector[K] d;           // treatments 
  array[N] vector[K] y;           // outcomes   

  // Covariates
  int<lower=0> P_x;
  array[N] matrix[K, P_x] X;    // K x P_xd

  // Nonlinear basis for each treatment. 
  int<lower=1> L;
  array[K] matrix[N, L] B;           // for each k: N × L, row n is basis at d[n,k]
}
parameters {
  matrix[K, M] A;
  matrix[K, M] gamma;

  vector[K] alpha_d;
  vector[K] alpha_y;

  real<lower=0> sigma_d;
  real<lower=0> sigma_y;

  // X effects
  // matrix[K, P_x] B_Xd;             // D ~ Xd
  // matrix[K, P_x] B_Xy;             // Y ~ Xy
  vector[P_x] B_Xd;            
  vector[P_x] B_Xy;             

  // Treatment effects: linear + nonlinear
  // matrix[K, K] beta_lin;
  real        beta_lin;           // linear part
  vector[L]   theta;              // spline coeffs (shared)

  // shrinkage for spline
  real<lower=0> tau_theta;
}
transformed parameters {
  vector<lower=0>[K] lambda1 = rep_vector(square(sigma_d), K);
  matrix[K, K] lambda1_inv = diag_matrix(1.0 ./ lambda1);
  cov_matrix[K] Sigma1_inv =
    lambda1_inv
    - lambda1_inv * A
      * inverse_spd(diag_matrix(rep_vector(1.0, M)) + A' * lambda1_inv * A)
      * A' * lambda1_inv;

  vector<lower=0>[K] lambda2 = rep_vector(square(sigma_y), K);
  matrix[K, K] lambda2_inv = diag_matrix(1.0 ./ lambda2);
  cov_matrix[K] Sigma2_inv =
    lambda2_inv
    - lambda2_inv * gamma
      * inverse_spd(diag_matrix(rep_vector(1.0, M)) + gamma' * lambda2_inv * gamma)
      * gamma' * lambda2_inv;
}
model {
  array[N] vector[K] mu;

  // Priors
  to_vector(A)       ~ normal(0, 1);
  to_vector(gamma)   ~ normal(0, 1);
  alpha_d            ~ normal(0, 2);
  alpha_y            ~ normal(0, 10);
  B_Xd               ~ normal(0, 1);
  B_Xy               ~ normal(0, 1);
  beta_lin           ~ normal(0, 5);
  theta              ~ normal(0, tau_theta);
  tau_theta          ~ normal(0, 1);
  sigma_d            ~ std_normal();
  sigma_y            ~ normal(0, 5);

  // bias mapping
  matrix[K, K] G;
  {
    matrix[M, M] mid = inverse_spd(diag_matrix(rep_vector(1.0, M)) - A' * Sigma1_inv * A);
    matrix[M, M] mid_sqrt = sqrtm(mid);
    G = gamma * mid_sqrt * A' * Sigma1_inv;     // K × K
  }

  // Likelihood
  for (n in 1:N) {
    // D model: alpha_d + X effects
    vector[K] x_eff_d = (P_x > 0) ? X[n] * B_Xd : rep_vector(0, K);
    d[n] ~ multi_normal_prec(alpha_d + x_eff_d, Sigma1_inv);

    // Y mean: intercept + X + bias(G d) + linear TE + nonlinear TE
    mu[n] = alpha_y;
    if (P_x > 0)
      mu[n] += X[n] * B_Xy;
    mu[n] += G * d[n] + beta_lin * d[n];
    // nonlinear TE, elementwise over k
    for (k in 1:K) {
      row_vector[L] b_row = B[k][n];             // 1 × L
      mu[n][k] += dot_product(theta, to_vector(b_row));
    }
  }

  y ~ multi_normal_prec(mu, Sigma2_inv);
}
generated quantities {
  // space for posterior predictions if you switch to sampling / laplace
}
