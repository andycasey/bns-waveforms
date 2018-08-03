
data {
  int<lower=1> D; // number of parameters
  int<lower=1> N; // number of waveforms
  real M1[N];     // mass of the primary
  real Lambda[N]; // dimensionless tidal deformability
  real y[N, D];   // the parameters to fit relations to

}

parameters {
  real<lower=0> sigma; // intrinsic sigma in the fit (over all parameters)
  matrix[2, D] a;      // hierarchical parameters for \alpha
  matrix[2, D] b;      // hierarchical parameters for \beta
}

transformed parameters {
  matrix[N, D] alpha_; // alpha
  matrix[N, D] beta_;  // beta

  // TODO: Matrix algebra you idiot
  for (n in 1:N) {
    for (d in 1:D) {
      alpha_[n, d] = a[1, d] * Lambda[n] + a[2, d];
      beta_[n, d] = b[1, d] * Lambda[n] + b[2, d];
    }
  }
}

model {
  sigma ~ beta(1, 5);
  for (n in 1:N)
    for (d in 1:D)
      y[n, d] ~ normal(alpha_[n, d] * pow(M1[n], beta_[n, d]), sigma);
}