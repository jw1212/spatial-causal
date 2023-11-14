// try to marginalize U out
data {
  int<lower=1> D;
  int<lower=1> N;
  array[N] vector[D] s;
  vector[N] t;
  vector[N] y;
}
transformed data {
  // vector[N] mu = rep_vector(0, N);
  real delta = 1e-8;
  // real eta2 = 0.9;
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0, upper=1> eta2;
  
  real a; // beta_T_tilde
  real b; // beta_U_tilde
  real<lower=0> sigma_t;
  real<lower=0> sigma_y;
  vector[N] z;
}
transformed parameters {
  real b_t = a - b / (1 - eta2) * eta2;
}
model {
  matrix[N, N] K = cov_exp_quad(s, alpha, rho);
  for (n in 1:N)
    K[n, n] = K[n, n] + delta;
  matrix[N, N] L_K = cholesky_decompose(K);
  
  // matrix[N, N] K_t = K;
  // real sq_sigma_t = square(sigma_t);
  // real sq_sigma_y = square(sigma_y);
  // for (n in 1:N)
  //   K_t[n, n] = K_t[n, n] + sq_sigma_t;
  // matrix[N, N] L_K_t = cholesky_decompose(K_t);
  // 
  // matrix[N, N] K_y = square(b) * K;
  // for (n in 1:N)
  //   K_y[n, n] = K_y[n, n] + sq_sigma_y;
  // matrix[N, N] L_K_y = cholesky_decompose(K_y);

  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma_t ~ std_normal();
  sigma_y ~ std_normal();
  z ~ std_normal();
  eta2 ~ beta(3, 3);
  
  vector[N] f = L_K * z;
  // f ~ multi_normal_cholesky(mu, L_K); // slow, don't converge?
  t ~ normal(f, sigma_t);
  y ~ normal(a * t + b * f, sigma_y);
  // t ~ multi_normal_cholesky(mu, L_K_t); // integrate out f and t,y as two GPs did not work
  // y ~ multi_normal_cholesky(a * t, L_K_y); // 870s for 7000 iterations, 4 times faster
}


