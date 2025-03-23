data {
    int<lower=1> N;
    array[N] real x;
    array[N] real y;
}

parameters {
    real alpha;
    real beta;
    real<lower=0> sigma2;
}

transformed parameters {
    real sigma;
    sigma = sqrt(sigma2);
}

model {
    // Priors
    alpha ~ normal(0, 10);
    beta ~ normal(0, 10);
    sigma2 ~ inv_gamma(1, 1);

    // Likelihood
    for (n in 1:N) {
        y[n] ~ normal(alpha + beta * x[n], sigma2);
    }
}