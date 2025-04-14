data {
    int<lower=1> N;
    int<lower=1> M;
    int<lower=1> K;
    matrix[N, K] x;
    matrix[M, K] x_test;
    vector[N] y;
}

parameters {
    real alpha;
    vector[K] beta;
    real<lower=0> sigma;
}

model {
    // Priors
    alpha ~ normal(0, 10);
    beta ~ multi_normal(zeros_vector(K), 10 * diag_matrix(rep_vector(1, K)));
    sigma ~ inv_gamma(2, 2);

    // Likelihood
    for (n in 1:N) {
        real i = 0;
        for (k in 1:K) {
            i += x[n, k] * beta[k];
        }
        y[n] ~ normal(alpha + i, sigma);
    }
}

generated quantities {
    vector[M] y_pred;
    for (m in 1:M) {
        real i = 0;
        for (k in 1:K) {
            i += x_test[m, k] * beta[k];
        }
        y_pred[m] = normal_rng(alpha + i, sigma);
    }
}