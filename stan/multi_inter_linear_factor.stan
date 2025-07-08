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
  int<lower=1> K; // number of treatments and outcomes
  int<lower=1> N; // number of observations
  int<lower=1> M; // number of confounders
  array[N] vector[K] d; // treatments
  array[N] vector[K] y; // outcomes
  array[N] vector[K] d_nb; // neighbor-average exposures  ( W %*% d )
  // matrix[K, K] D; // distances
}
parameters {
  matrix[K, M] A;
  matrix[K, M] gamma;
  
  // real<lower=0> lambda_mean; // mean for all lambda
  // vector<lower=0>[K] lambda1; // lambda for T~U
  // vector<lower=0>[K] lambda2; // lambda for Y~T,U
  real<lower=0> sigma_d;
  real<lower=0> sigma_y;
  
  real beta1; // direct treatment effect
  real beta2; // interference effect
  real<lower=0> alpha; 
  // intercepts
  vector[K] alpha_d;
  vector[K] alpha_y;
}
transformed parameters {
  vector<lower=0>[K] lambda1 = rep_vector(square(sigma_d), K);
  matrix[K, K] lambda1_inv = diag_matrix(1.0 ./ lambda1);
  cov_matrix[K] Sigma1_inv = lambda1_inv - lambda1_inv * A 
  * inverse_spd(diag_matrix(rep_vector(1, M)) +  A' * lambda1_inv * A) * A' * lambda1_inv;
  
  vector<lower=0>[K] lambda2 = rep_vector(square(sigma_y), K);
  matrix[K, K] lambda2_inv = diag_matrix(1.0 ./ lambda2);
  cov_matrix[K] Sigma2_inv = lambda2_inv - lambda2_inv * gamma 
  * inverse_spd(diag_matrix(rep_vector(1, M)) +  gamma' * lambda2_inv * gamma) * gamma' * lambda2_inv;
}
model {
  array[N] vector[K] mu;
  // array[N] vector[K] interference;
  
  for(k in 1:K) {
    for(m in 1:M) {
      A[k, m] ~ normal(0, 1);
      gamma[k, m] ~ normal(0, 1);
    }
  }
  
  // // Compute interference term for each observation
  // for (n in 1:N) {
  //   interference[n] = rep_vector(0, K);
  //   for (i in 1:K) {
  //     for (j in 1:K) {
  //       if (i != j) {
  //         real W_ij = exp(-alpha * D[i, j]);
  //         interference[n][i] += W_ij * d[n][j];  // sum weighted treatments of other units
  //       }
  //     }
  //   }
  // }
  
  beta1 ~ normal(0, 5);
  beta2 ~ normal(0, 5);
  alpha_d ~ normal(0, 10);
  alpha_y ~ normal(0, 10);
  sigma_d ~ std_normal();
  sigma_y ~ std_normal();
  // lambda_mean ~ cauchy(0, 1);
  // lambda1 ~ lognormal(log(lambda_mean), 1);
  // lambda2 ~ lognormal(log(lambda_mean), 1);
  
  for (n in 1:N) {
    d[n] ~ multi_normal_prec(alpha_d, Sigma1_inv);
    mu[n] = (gamma * sqrtm(inverse_spd(diag_matrix(rep_vector(1, M)) - A' * Sigma1_inv * A)) * A' * Sigma1_inv
    + diag_matrix(rep_vector(beta1, K))) * d[n] + diag_matrix(rep_vector(beta2, K)) * d_nb[n]; // model bias explicitly
  }
    
  y ~ multi_normal_prec(mu, Sigma2_inv);  // condition on t=d
}
