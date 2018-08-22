
// Linear model for binary neutron star waveforms

data {
  int<lower=1> F; // number of frequency points
  int<lower=1> N; // number of waveforms
  int<lower=1> P; // number of parameters
  matrix[N, F] y;
  matrix[N, P] whitened_labels;
}

parameters {
  vector<lower=0>[F] sigma; // intrinsic scatter at each frequency
  matrix[F, 1 + P] theta;
}

transformed parameters {
  matrix[N, 1 + P] DM;

  DM = rep_matrix(1.0, N, 1 + P);
  for (n in 1:N)
    for (p in 1:P)
      DM[n, 1 + p] = whitened_labels[n, p];

}
model {
  for (f in 1:F)
    for (p in 1:P)
      theta[f, p] ~ normal(0, 1);
      
  for (i in 1:N)
    y[i] ~ normal(theta * DM[i]', sigma);
}