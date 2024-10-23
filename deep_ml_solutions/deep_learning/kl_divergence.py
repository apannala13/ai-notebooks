import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
  # log(σ_q / σ_p)
  term_one = np.log(sigma_q / sigma_p)
  # (σ_p^2 + (μ_p - μ_q)^2) / (2 * σ_q^2) 
  term_two = (np.square(sigma_p) + np.square(mu_p - mu_q)) / (2 * np.square(sigma_q))
  divergence = term_one + term_two - 0.5 
  return divergence
