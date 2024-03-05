data {
  int<lower=1> D;
  int<lower=1> N;
  array[N] vector[D] s;
  vector[N] t;
  vector[N] y;
}
transformed data {
  vector[N] mu = rep_vector(0, N);
  // real delta = 1e-8;
  // real etaT2 = 0.5;
  // real etaY2 = 1.0 / 3;
}
parameters {
  real<lower=0> rho_t;
  real<lower=0> alpha_t;
  real<lower=0> rho_y;
  real<lower=0> alpha_y;
  
  real b_t; // beta_T_tilde
  real<lower=0> sigma_t;
  real<lower=0> sigma_y;
  
  // real<lower=0, upper=1> etaT2;
  // real<lower=0, upper=1> etaY2;
}
model {
  matrix[N, N] K_t = gp_exp_quad_cov(s, alpha_t, rho_t);
  real sq_sigma_t = square(sigma_t);
  for (n in 1:N)
    K_t[n, n] = K_t[n, n] + sq_sigma_t;
  matrix[N, N] L_K_t = cholesky_decompose(K_t);
  
  matrix[N, N] K_y = gp_exp_quad_cov(s, alpha_y, rho_y);
  real sq_sigma_y = square(sigma_y);
  for (n in 1:N)
    K_y[n, n] = K_y[n, n] + sq_sigma_y;
  matrix[N, N] L_K_y = cholesky_decompose(K_y);

  rho_t ~ inv_gamma(5, 5);
  alpha_t ~ std_normal();
  sigma_t ~ std_normal();
  rho_y ~ inv_gamma(5, 5);
  alpha_y ~ std_normal();
  sigma_y ~ std_normal();
  
  t ~ multi_normal_cholesky(mu, L_K_t);
  y ~ multi_normal_cholesky(b_t * t, L_K_y);
}
// generated quantities {
//   real b_t_lower = b_t - sqrt(sigma_y / sigma_t / (1 - etaT2) * etaT2 * etaY2);
//   real b_t_upper = b_t + sqrt(sigma_y / sigma_t / (1 - etaT2) * etaT2 * etaY2);
// }
