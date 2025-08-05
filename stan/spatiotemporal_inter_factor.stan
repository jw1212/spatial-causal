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
  int<lower=1> K; // number of treatments (including neighborhood)
  int<lower=1> Q; // number of outcomes
  int<lower=1> N; // number of observations
  int<lower=1> M; // number of confounders
  array[N] vector[K] d; // treatments
  array[N] vector[K] d_nb; // neighborhood treatments
  array[N] vector[Q] y; // outcomes
}
parameters {
  matrix[K, M] A;
  matrix[Q, M] gamma;
  
  // real<lower=0> lambda_mean; // mean for all lambda
  // vector<lower=0>[K] lambda1; // lambda for T~U
  // vector<lower=0>[Q] lambda2; // lambda for Y~T,U
  real<lower=0> sigma_d;
  real<lower=0> sigma_y;
  
  real beta0; // biased effect for the first outcome
  real beta1; // direct effect
  real beta2; // temporal interference effect
  real beta3; // neighborhood interference effect
  vector[K] alpha_d; // intercept
  vector[Q] alpha_y; // intercept
}
model {
  vector[K] lambda1 = rep_vector(square(sigma_d), K);
  matrix[K, K] lambda1_inv = diag_matrix(1.0 ./ lambda1); 
  matrix[K, K] Sigma1_inv = lambda1_inv - lambda1_inv * A 
  * inverse_spd(diag_matrix(rep_vector(1, M)) +  A' * lambda1_inv * A) * A' * lambda1_inv; //  + diag_matrix(rep_vector(1e-6, K))
  
  vector[Q] lambda2 = rep_vector(square(sigma_y), Q);
  matrix[Q, Q] lambda2_inv = diag_matrix(1.0 ./ lambda2); 
  matrix[Q, Q] Sigma2_inv = lambda2_inv - lambda2_inv * gamma 
  * inverse_spd(diag_matrix(rep_vector(1, M)) +  gamma' * lambda2_inv * gamma) * gamma' * lambda2_inv;
 
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
  }
  
  beta0 ~ normal(0, 5);
  beta1 ~ normal(0, 5);
  beta2 ~ normal(0, 2);
  beta3 ~ normal(0, 2);
  alpha_d ~ normal(0, 10);
  alpha_y ~ normal(0, 10);
  sigma_d ~ normal(0, 2);
  sigma_y ~ normal(0, 10);
  // lambda_mean ~ cauchy(0, 1);
  // lambda1 ~ lognormal(log(lambda_mean), 1);
  // lambda2 ~ lognormal(log(lambda_mean), 1);
  
  matrix[Q, K] bigC;
  bigC = gamma * sqrtm(inverse_spd(diag_matrix(rep_vector(1, M)) - A' * Sigma1_inv * A)) * A' * Sigma1_inv;
  for (q in 1:Q) {
    if (q == 1) {
      bigC[q, q] += beta0;
    } else {
      bigC[q, q]     += beta1;
      bigC[q, q - 1] += beta2;
    }
  }
  
  for (n in 1:N)
    mu[n] = alpha_y + bigC * d[n] + beta3 * d_nb[n]; // model bias explicitly
  
  d ~ multi_normal_prec(alpha_d, Sigma1_inv);
  y ~ multi_normal_prec(mu, Sigma2_inv);  // condition on t=d
}

