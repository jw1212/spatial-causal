data {
  int<lower=1> D; // dim of location
  int<lower=1> N; // number of locations
  int<lower=1> T0; // number of time points 
  array[N] vector[D] s;
  array[T0] vector[N] d;
  array[T0] vector[N] y;
}
transformed data {
  vector[N] mu_d = rep_vector(0, N);
  // real delta = 1e-8;
  // real etaT2 = 0.5;
  // real etaY2 = 1.0 / 3;
}
parameters {
  // real<lower=0> rho_d;
  // real<lower=0> alpha_d;
  real<lower=0> rho_y;
  real<lower=0> alpha_y;
  
  // real beta; // beta_T_tilde
  // matrix[N, N] B; // naive estimate
  real beta;
  // real<lower=0> lambda_mean; // mean for all lambda
  // vector<lower=0>[N] lambda1; // lambda for T~U
  // vector<lower=0>[N] lambda2; // lambda for Y~T,U
  // real<lower=0> sigma_d;
  real<lower=0> sigma_y;
  
  // real<lower=0, upper=1> etaD2;
  // real<lower=0, upper=1> etaY2;
}
transformed parameters {
  // matrix[N, N] K_d = gp_exp_quad_cov(s, alpha_d, rho_d);
  // // real sq_sigma_d = square(sigma_d);
  // for (n in 1:N)
  //   K_d[n, n] = K_d[n, n] + lambda1[n]; // sq_sigma_d;
  // matrix[N, N] L_K_d = cholesky_decompose(K_d);
  real sq_sigma_y = square(sigma_y);
  
}
model {
  matrix[N, N] K_y = gp_exp_quad_cov(s, alpha_y, rho_y);
  // real sq_sigma_y = square(sigma_y);
  for (n in 1:N)
    K_y[n, n] = K_y[n, n] + sq_sigma_y; // lambda2[n];
  matrix[N, N] L_K_y = cholesky_decompose(K_y);
  
  beta ~ normal(0, 5);
  
  // rho_d ~ inv_gamma(5, 5);
  // alpha_d ~ std_normal();
  rho_y ~ inv_gamma(5, 5);
  alpha_y ~ std_normal();
  // sigma_d ~ std_normal();
  sigma_y ~ std_normal();
  // lambda_mean ~ cauchy(0, 1);
  // lambda1 ~ lognormal(log(lambda_mean), 1);
  // lambda2 ~ lognormal(log(lambda_mean), 1);
  
  array[T0] vector[N] mu;
  for (i in 1:T0) {
    mu[i] = diag_matrix(rep_vector(beta, N)) * d[i];
    // d[i] ~ multi_normal_cholesky(mu_d, L_K_d);
    y[i] ~ multi_normal_cholesky(mu[i], L_K_y);
  }
}
