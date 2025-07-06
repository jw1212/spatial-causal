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
  int<lower=1> K; // number of treatments
  int<lower=1> Q; // number of outcomes
  int<lower=1> N; // number of observations
  int<lower=1> M; // number of confounders
  array[N] vector[K] d; // treatments
  array[N] vector[Q] y; // outcomes
}
transformed data {
  array[N * K] real d_flat;     // Flatten the array into a new array
  for (n in 1:N) {
    for (k in 1:K) {
      d_flat[(n - 1) * K + k] = d[n][k];  // Copy elements to flattened array
    }
  }
}
parameters {
  matrix[K, M] A;
  real<lower=0> sigma_d;
  matrix[Q, M] gamma;
  real<lower=0> sigma_y;

  real<lower=0> alpha_d;    // Output scale for the GP on treatment effect
  real<lower=0> rho_d;      // Length scale for the GP on treatment effect
  vector[N * K] eta;
}
transformed parameters {
  matrix[N * K, N * K] K_d = gp_exp_quad_cov(d_flat, alpha_d, rho_d) + diag_matrix(rep_vector(1e-6, N*K));
  matrix[N * K, N * K] L_K_d = cholesky_decompose(K_d);
  vector[N * K] f_d = L_K_d * eta;
  
  vector<lower=0>[K] lambda1 = rep_vector(square(sigma_d), K);
  matrix[K, K] lambda1_inv = diag_matrix(1.0 ./ lambda1); 
  cov_matrix[K] Sigma1_inv = lambda1_inv - lambda1_inv * A 
  * inverse_spd(diag_matrix(rep_vector(1, M)) +  A' * lambda1_inv * A) * A' * lambda1_inv;
  
  vector<lower=0>[Q] lambda2 = rep_vector(square(sigma_y), Q);
  matrix[Q, Q] lambda2_inv = diag_matrix(1.0 ./ lambda2); 
  cov_matrix[Q] Sigma2_inv = lambda2_inv - lambda2_inv * gamma 
  * inverse_spd(diag_matrix(rep_vector(1, M)) +  gamma' * lambda2_inv * gamma) * gamma' * lambda2_inv;
}
model {
  array[N] vector[Q] mu;
  
  for(k in 1:K) {
    for(m in 1:M) {
      A[k, m] ~ normal(0, 1);
    }
  }
  
  for(q in 1:Q) {
    for(m in 1:M) {
      gamma[q, m] ~ normal(0, 1);
    }
    
    // for(k in 1:K) {
    //   B[q, k] ~ normal(0, 1);
    // }
  }
  
  alpha_d ~ std_normal();
  rho_d ~ inv_gamma(5, 5);
  sigma_d ~ std_normal();
  sigma_y ~ std_normal();
  // lambda_mean ~ cauchy(0, 1);
  // lambda1 ~ lognormal(log(lambda_mean), 1);
  // lambda2 ~ lognormal(log(lambda_mean), 1);
  
  for (n in 1:N)
    // mu[n] = beta + A * d[n];
    mu[n] = gamma * sqrtm(inverse_spd(diag_matrix(rep_vector(1, M)) - A' * Sigma1_inv * A)) * A' * Sigma1_inv * d[n] + f_d[((n - 1) * K + 1):n*K]; // model bias explicitly
  
  d ~ multi_normal_prec(rep_vector(0, K), Sigma1_inv);
  y ~ multi_normal_prec(mu, Sigma2_inv);  // condition on t=d
}
