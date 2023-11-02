data {
  int<lower=1> D;
  int<lower=1> N;
  array[N] vector[D] s;
  vector[N] t;
}
transformed data {
  vector[N] mu = rep_vector(0, N);
}
parameters {
  real<lower=0> rho_f;
  real<lower=0> alpha_f;
  real<lower=0> sigma_t;
  
  // real a; // beta_T_tilde
  // real b; // beta_U_tilde
  // real c;
  // real<lower=0> sigma_x;
  // real<lower=0> sigma_y;
}
model {
  matrix[N, N] K_f = gp_exp_quad_cov(s, alpha_f, rho_f);
  real sq_sigma_t= square(sigma_t);
  for (n in 1:N)
    K_f[n, n] = K_f[n, n] + sq_sigma_t;
  matrix[N, N] L_K_f = cholesky_decompose(K_f);
  
  rho_f ~ inv_gamma(5, 5);
  alpha_f ~ std_normal();
  sigma_t ~ std_normal();

  t ~ multi_normal_cholesky(mu, L_K_f);
  // f0 ~ std_normal();
  // vector[N] f = L_K_f * f0;
  // t ~ normal(f, sigma_t);
}
