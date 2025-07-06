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
parameters {
  matrix[K, M] A;
  matrix[Q, M] gamma;
  vector[Q] alpha_y; // intercepts for Y
  vector[K] alpha_d; // intercepts for D
  
  // real<lower=0> lambda_mean; // mean for all lambda
  // vector<lower=0>[K] lambda1; // lambda for T~U
  // vector<lower=0>[Q] lambda2; // lambda for Y~T,U
  real<lower=0> sigma_d;
  real<lower=0> sigma_y;
  
  real beta; // treatment effect
  // second parameterization
  // matrix[M, Q] A;
  // vector[Q] beta; // intercept
  // matrix[Q, K] B; // naive estimate
}
transformed parameters {
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
  
  beta ~ normal(0, 5);
  alpha_d ~ normal(0, 10);
  alpha_y ~ normal(0, 10);
  sigma_d ~ std_normal();
  sigma_y ~ std_normal();
  // lambda_mean ~ cauchy(0, 1);
  // lambda1 ~ lognormal(log(lambda_mean), 1);
  // lambda2 ~ lognormal(log(lambda_mean), 1);
  
  for (n in 1:N) {
    d[n] ~ multi_normal_prec(alpha_d, Sigma1_inv);
    mu[n] = alpha_y + (gamma * sqrtm(inverse_spd(diag_matrix(rep_vector(1, M)) - A' * Sigma1_inv * A)) * A' * Sigma1_inv
    + diag_matrix(rep_vector(beta, Q))) * d[n]; // model bias explicitly
  }
  y ~ multi_normal_prec(mu, Sigma2_inv);  // condition on d
}


