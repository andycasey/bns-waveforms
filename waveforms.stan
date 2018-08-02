data {
  int<lower=1> D; // number of dimensions
  int<lower=1> N; // number of waveforms
  int<lower=1> L; // number of labels
  matrix[N, D] y;
  matrix[N, L] white_labels;
}


parameters {
  vector<lower=0>[D] sigma;
  matrix[D, 1 + L] theta;
}

transformed parameters {
  matrix[N, 1 + L] DM;

  DM = rep_matrix(1.0, N, 1 + L);
  for (n in 1:N)
    for (l in 1:L) {
      DM[n, 1 + l] = white_labels[n, l];
    }
}
model {
  for (i in 1:N)
    y[i] ~ normal(theta * DM[i]', sigma);
}