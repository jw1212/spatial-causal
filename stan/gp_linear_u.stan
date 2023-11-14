// data {
//   int<lower=1> D;
//   int<lower=1> N;
//   array[N] vector[D] s;
//   vector[N] t;
//   vector[N] y;
// }
// transformed data {
//   vector[N] mu = rep_vector(0, N);
//   real sq_sigma_u = 0.2; // treat sigma2 fixed
// }
// parameters {
//   real<lower=0> rho_u;
//   real<lower=0> alpha_u;
//   // real<lower=0> sigma_u; // treat sigma2 random
// 
//   vector[N] u;
//   real b_t;
//   real b_u;
//   real<lower=0> sigma_t;
//   real<lower=0> sigma_y;
// }
// model {
//   matrix[N, N] K_u = cov_exp_quad(s, alpha_u, rho_u);
//   // real sq_sigma_u = square(sigma_u);
//   // diagonal elements
//   for (n in 1:N)
//     K_u[n, n] = K_u[n, n] + sq_sigma_u;
//   matrix[N, N] L_K_u = cholesky_decompose(K_u);
// 
//   rho_u ~ inv_gamma(5, 5);
//   alpha_u ~ std_normal();
//   // sigma_u ~ std_normal();
//   sigma_t ~ std_normal();
//   sigma_y ~ std_normal();
// 
//   u ~ multi_normal_cholesky(mu, L_K_u);
//   t ~ normal(u, sigma_t);
//   y ~ normal(b_t * t + b_u * u, sigma_y);
// }

// assign prior on eta2 and beta_T instead of sigma2 and sigmat2

data {
  int<lower=1> D;
  int<lower=1> N;
  array[N] vector[D] s;
  vector[N] t;
  vector[N] y;
}
transformed data {
  vector[N] mu = rep_vector(0, N);
  real delta = 1e-8;
  // real eta2 = 0.9; // treat eta2 fixed
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0, upper=0.6> eta2; // treat eta2 random

  vector[N] z;
  real b_t; // beta_T
  real b_u; // beta_U_tilde
  real<lower=0> sigma_t;
  real<lower=0> sigma_y;
}
model {
  matrix[N, N] K = cov_exp_quad(s, alpha, rho);
  for (n in 1:N)
    K[n, n] = K[n, n] + delta;
  matrix[N, N] L_K = cholesky_decompose(K);

  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma_t ~ std_normal();
  sigma_y ~ std_normal();
  z ~ std_normal();
  // eta2 ~ beta(3, 3);

  vector[N] f = L_K * z;
  t ~ normal(f, sigma_t);
  y ~ normal((b_t + b_u / (1 - eta2) * eta2) * t + b_u * f, sigma_y);
}
