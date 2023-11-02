functions {
  matrix kronecker_prod(matrix A, matrix B) {
    matrix[rows(A) * rows(B), cols(A) * cols(B)] C;
    int m;
    int n;
    int p;
    int q;
    m = rows(A);
    n = cols(A);
    p = rows(B);
    q = cols(B);
    for (i in 1:m) {
      for (j in 1:n) {
        int row_start;
        int row_end;
        int col_start;
        int col_end;
        row_start = (i - 1) * p + 1;
        row_end = (i - 1) * p + p;
        col_start = (j - 1) * q + 1;
        col_end = (j - 1) * q + q;
        C[row_start:row_end, col_start:col_end] = A[i, j] * B;
      }
    }
    return C;
  }
}
data {
  int<lower=1> D;
  int<lower=1> M;
  int<lower=1> N;
  array[M] vector[D] s;
  // vector[M] t0;
  // array[N] vector[D] s;
  // vector[M] h_unique;
  vector[N] t;
  vector[N] y;
}
transformed data {
  real delta = 1e-9;
}
parameters {
  // real<lower=0> rho_f;
  // real<lower=0> alpha_f;
  // real<lower=0> sigma_u;
  
  // real<lower=0> rho_z;
  // real<lower=0> alpha_z;
  
  real<lower=0> rho_w;
  real<lower=0> alpha_w;
  
  // vector[M] f0;
  // vector[M] z0;
  // vector[M] u;
  vector[M] w0;
  real beta; // beta_T_tilde
  // real b; // beta_U_tilde
  // real<lower=0> sigma_t;
  // real<lower=0> sigma_y_tilde;
  real<lower=0> sigma_y;
  real<lower=sigma_y> sigma_y_tilde;
}
transformed parameters{
  real sq_sigma_y = square(sigma_y);
  real sq_sigma_y_tilde =  square(sigma_y_tilde);
}
model {
  // need to add noise and intraclass correlation into the GP kernel
  int r = N %/% M;
  matrix[M, M] K_w = gp_exp_quad_cov(s, alpha_w, rho_w);
  for (i in 1:M)
    K_w[i, i] = K_w[i, i] + delta;
  matrix[M, M] L_K_w = cholesky_decompose(K_w);
  
  matrix[r, r] K_r = rep_matrix(sq_sigma_y_tilde - sq_sigma_y, r, r);
  for (i in 1:r)
    K_r[i, i] = K_r[i, i] + sq_sigma_y;
  matrix[N, N] K_y = kronecker_prod(diag_matrix(rep_vector(1, M)), K_r);
  matrix[N, N] L_K_y = cholesky_decompose(K_y);
  
  rho_w ~ inv_gamma(5, 5);
  alpha_w ~ std_normal();
  // sigma_y_tilde ~ std_normal();
  sigma_y ~ std_normal();
  sigma_y_tilde ~ std_normal();
  w0 ~ std_normal();

  // vector[N] w = L_K_w * w0;
  
  // t0 ~ normal(f, sigma_t);
  // u ~ normal(0, 1);
  // vector[M] u1 = u;
  // vector[N] u1 = append_row(u, append_row(u, append_row(u, append_row(u, u))));
  // y ~ normal(a * t + w + eps_tilde, sigma_y);
  // h_unique ~ multi_normal_cholesky(rep_vector(0, M), L_K_w);
  vector[M] h_unique = L_K_w * w0;
  vector[N] h;
  for (i in 1:M) 
    h[((i - 1) * r + 1): ((i - 1) * r + r)] = h_unique[i] * rep_vector(1, r);
  y ~ multi_normal_cholesky(beta * t + h, L_K_y);
}
// generated quantities {
//   int r = N %/% M;
//   matrix[M, M] K_r = rep_matrix(sq_sigma_y_tilde - sq_sigma_y, M, M);
//   for (i in 1:M)
//     K_r[i, i] = K_r[i, i] + sq_sigma_y;
//   matrix[N, N] K_y = kronecker_prod(diag_matrix(rep_vector(1, r)), K_r);
// }
