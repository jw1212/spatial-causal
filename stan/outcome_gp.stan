data {
  int<lower=1> D;
  int<lower=1> N;
  array[N] vector[D] s;
  vector[N] t;
  vector[N] y;
}
// transformed data {
//   real delta = 1e-8;
// }
parameters {
  real<lower=0> rho_w;
  real<lower=0> alpha_w;
  real<lower=0> sigma_y;
  
  // vector[N] w0;
  real beta; // beta_T_tilde
  // real b; // beta_U_tilde
  // real<lower=0> sigma_x;
  // real<lower=0> sigma_y;
}
transformed parameters{
  real sq_sigma_y = square(sigma_y);
}
model {
  matrix[N, N] K_w = gp_exp_quad_cov(s, alpha_w, rho_w);
  for (n in 1:N)
    K_w[n, n] = K_w[n, n] + sq_sigma_y;
  matrix[N, N] L_K_w = cholesky_decompose(K_w);
  
  rho_w ~ inv_gamma(5, 5);
  alpha_w ~ std_normal();
  sigma_y ~ std_normal();

  // w0 ~ std_normal();
  // vector[N] w = L_K_w * w0;
  // y ~ normal(a * t + w, sigma_y);
  y ~ multi_normal_cholesky(beta * t, L_K_w);
}
