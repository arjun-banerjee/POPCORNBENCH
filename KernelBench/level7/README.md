# Probabilistic and Statistical Model reference problems

These modules implement compute-intensive kernels from **probabilistic inference**, **Monte Carlo methods**, **Bayesian models**, and **statistical machine learning**. They cover patterns such as MCMC sampling steps, Gaussian process regression, variational inference, normalizing flows, particle filters, and hidden Markov models.

Each file is a self-contained PyTorch reference (`class Model`, `get_inputs()`, `get_init_inputs()`). They are **not** wired into the default KernelBench HuggingFace dataset; use `ref_origin=local` and point to this directory.

## Suggested mapping to probabilistic domains

| File | Domain | Typical use |
|------|--------|-------------|
| `1_MetropolisHastingsStep.py` | MCMC | Single MH accept/reject step for posterior sampling |
| `2_HamiltonianMonteCarloStep.py` | MCMC | Leapfrog-integration HMC proposal + accept/reject |
| `3_GibbsSamplingStep.py` | MCMC | Conditional sampling from full conditionals |
| `4_GaussianProcessRBFKernel.py` | Gaussian Processes | RBF covariance matrix computation |
| `5_GaussianProcessRegression.py` | Gaussian Processes | GP posterior predictive (Cholesky solve) |
| `6_VariationalELBO.py` | Variational Inference | Evidence lower bound for VAEs |
| `7_ReparameterizationTrick.py` | Variational Inference | Differentiable sampling via reparameterization |
| `8_SteinVariationalGradient.py` | Particle VI | SVGD kernel + gradient update |
| `9_ImportanceSampling.py` | Monte Carlo | Self-normalized importance weights + estimate |
| `10_ParticleFilter.py` | Sequential MC | Bootstrap particle filter (predict–weight–resample) |
| `11_BayesianLinearRegression.py` | Bayesian Models | Posterior parameter update with conjugate prior |
| `12_DirichletMultinomialLogLikelihood.py` | Bayesian Models | Log-likelihood under Dirichlet-Multinomial |
| `13_HiddenMarkovForward.py` | Graphical Models | Forward algorithm for HMM log-likelihood |
| `14_GaussianMixtureModelEM.py` | Mixture Models | Single E-step + M-step of EM for GMM |
| `15_NormalizingFlowPlanar.py` | Normalizing Flows | Planar flow layer with log-det-Jacobian |
| `16_ConditionalVAELoss.py` | Deep Generative | Conditional VAE reconstruction + KL loss |
