// marginalize U out
data {
  int<lower=1> D;
  int<lower=1> N;
  // int<lower=1> P;
  array[N] vector[D] s;
  // matrix[N, P] x;    // Covariates
  vector[N] t;
  vector[N] y;
}
transformed data {
  // vector[N] mu = rep_vector(0, N);
  real delta = 1e-8;
  // real eta2 = 0.5;
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  // real<lower=0, upper=1> eta2;
  real beta_t0;             // Intercept for treatment model
  real beta_y0;             // Intercept for outcome model
  // vector[P] gamma_t;        // Effect of X on treatment
  // vector[P] gamma_y;        // Effect of X on outcome
  
  real a; // beta_T_tilde
  real b; // beta_U_tilde
  real<lower=0> sigma_t;
  real<lower=0> sigma_y;
  vector[N] z;
}
// transformed parameters {
//   // Compute the covariance matrix using the exponentiated quadratic function
//   matrix[N, N] K = gp_exp_quad_cov(s, alpha, rho);
//   // Add jitter to the diagonal for stability
//   K += diag_matrix(rep_vector(delta, N));
//   
//   // Compute the Cholesky factor
//   matrix[N, N] L_K = cholesky_decompose(K);
//   // Non-centered parameterization: compute the latent function values
//   vector[N] f = L_K * z;
// }
model {
  // not identify sign of b
  // matrix[N, N] K_t = gp_exp_quad_cov(s, alpha, rho);
  // matrix[N, N] K_y = square(b) * K_t;
  // real sq_sigma_t = square(sigma_t);
  // for (n in 1:N)
  //   K_t[n, n] = K_t[n, n] + sq_sigma_t;
  // matrix[N, N] L_K_t = cholesky_decompose(K_t);
  // 
  // real sq_sigma_y = square(sigma_y);
  // for (n in 1:N)
  //   K_y[n, n] = K_y[n, n] + sq_sigma_y;
  // matrix[N, N] L_K_y = cholesky_decompose(K_y);
  
  matrix[N, N] K = gp_exp_quad_cov(s, alpha, rho);
  for (n in 1:N)
    K[n, n] = K[n, n] + delta;
  matrix[N, N] L_K = cholesky_decompose(K);

  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma_t ~ std_normal();
  sigma_y ~ normal(0, 10); 
  a ~ normal(0, 5); 
  b ~ normal(0, 5); 
  
  beta_t0 ~ normal(0, 1); 
  beta_y0 ~ normal(0, 1);
  // gamma_t ~ normal(0, 5); 
  // gamma_y ~ normal(0, 5); 
  // 
  z ~ std_normal();
  // eta2 ~ beta(3, 3);
  
  vector[N] f = L_K * z;
  t ~ normal(beta_t0 + f, sigma_t); // + x * gamma_t
  y ~ normal(beta_y0 + a * t + b * f, sigma_y); // + x * gamma_y
  
  // t ~ multi_normal_cholesky(mu, L_K_t);
  // y ~ multi_normal_cholesky(a * t, L_K_y);
}
// generated quantities {
//   real b_t = a - b / (1 - eta2) * eta2;
// }

