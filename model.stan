data {
  int<lower=1> D; // number of dimensions
  int<lower=1> N; // number of waveforms
  int<lower=1> M; // number of equations of state
  real x[N];
  real y[N, D];
  int eos_index[N];
}

parameters {
  real<lower=0> sigma;
  real<lower=0, upper=1> eos_coeff[M];
  real alpha_slope[D];
  real alpha_offset[D];
  real beta_slope[D];
  real beta_offset[D];
}

transformed parameters {
  real a[M, D];
  real b[M, D];
  for (i in 1:M) {
    for (d in 1:D) {
      a[i, d] = alpha_slope[d] * eos_coeff[i] + alpha_offset[d];
      b[i, d] = beta_slope[d] * eos_coeff[i] + beta_offset[d];
    }
  }
}

model {
  sigma ~ beta(1, 5);
  for (i in 1:N)
    for (d in 1:D)
      y[i, d] ~ normal(a[eos_index[i], d] * pow(x[i], b[eos_index[i], d]), sigma);
}