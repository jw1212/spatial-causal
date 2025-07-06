data {
  int<lower=1> D; // spatial dimension
  int<lower=1> N; // number of locations
  int<lower=1> T0; // number of time points
  int<lower=1> M;  // number of inducing points
  // int<lower=1> P;  // number of covariates, optional
  
  array[N] vector[D] s;
  array[M] vector[D] s0;    // coordinates for inducing sites
  array[T0] vector[N] d;
  array[T0] vector[N] y;
  // array[T0] matrix[N, P] X;           // covariate matrix, optional
}
transformed data {
  real jitter = 1e-8;
  matrix[N, M] dist2_nm;
  matrix[M, M] dist2_mm;

  for (i in 1:N)
    for (j in 1:M)
      dist2_nm[i, j] = squared_distance(s[i], s0[j]);

  for (j1 in 1:M)
    for (j2 in j1:M) {
      real d2 = squared_distance(s0[j1], s0[j2]);
      dist2_mm[j1, j2] = d2;
      dist2_mm[j2, j1] = d2;
    }
}
parameters {
  real<lower=0> rho;
  real<lower=0> alpha;
  vector[M] eta_raw;
  
  real beta_d0;             // Intercept for treatment model
  real beta_y0;             // Intercept for outcome model
  // vector[P] beta_d;          // coefficients for X in treatment model, optional
  // vector[P] beta_y;          // coefficients for X in outcome model, optional
  real a; // beta_T_tilde
  real b; // beta_U_tilde
  real<lower=0> sigma_d;
  real<lower=0> sigma_y;
}
transformed parameters {
  /* 1.  kernel matrices for inducing points */
  matrix[M, M] K_mm =
      square(alpha) .* exp(-0.5 * dist2_mm / square(rho));
  for (m in 1:M) K_mm[m, m] += jitter;        // stabilise

  matrix[M, M] L_mm = cholesky_decompose(K_mm);

  /* 2.  draw latent GP at inducing sites: f0 ~ N(0, K_mm) */
  vector[M] f0 = L_mm * eta_raw;

  /* 3.  project to all N sites: f = K_nm K_mm⁻¹ f0  */
  matrix[N, M] K_nm =
      square(alpha) .* exp(-0.5 * dist2_nm / square(rho));

  /* cholesky_solve avoids explicit inverse: solves K_mm * x = f0 */
  vector[M] v  = mdivide_left_tri_low(L_mm, f0);   // solve  L_mm  v = f0
  vector[M] u  = mdivide_left_tri_low(L_mm', v);    // solve  L_mm' u = v
  // vector[M] u = cholesky_solve(L_mm, f0);
  vector[N] f = K_nm * u;
}
model {
  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  eta_raw  ~ std_normal();
  
  sigma_d ~ normal(0, 2);
  sigma_y ~ normal(0, 10); 
  // beta_d ~ normal(0, 2);
  // beta_y ~ normal(0, 5);
  a ~ normal(0, 1); 
  b ~ normal(0, 10); 
  beta_d0 ~ normal(0, 2); 
  beta_y0 ~ normal(0, 10);
  
  for (i in 1:T0) {
    d[i] ~ normal(beta_d0 + f, sigma_d); // + X[i] * beta_d, X is optional
    y[i] ~ normal(beta_y0 + a * d[i] + b * f, sigma_y); // + X[i] * beta_y
  }
}
