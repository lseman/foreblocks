# Zero-cost metric contracts

These scores rank candidates within a search run. They are not estimates of
forecast error. Compare candidates with the same input batch, initialization
policy, output shape, and metric configuration. The weighted aggregate is a
heuristic; its weights require calibration on held-out, fully trained models.

| Metric | Implemented quantity | Scope / limitation |
| --- | --- | --- |
| `params` | Unique trainable parameter count | Exact for the instantiated model, including all supernet branches. |
| `flops` | Two operations per multiply-add in Conv and Linear layers, per example | Partial count. Attention matrix products, normalization, activation functions, and data movement are excluded. |
| `synflow` | Sum of absolute parameter-gradient products after weight linearization on ones input | Uses the candidate's current weights; run at initialization for original SynFlow conditions. |
| `snip` | Sum of absolute parameter-gradient products for named weights | `current` is a trained-weight variant. `init` resets parameters before scoring. |
| `grasp` | Negative sum of parameter times Hessian-gradient product | Expensive second-order proxy; backend fallback may fail. |
| `fisher` | Sum of squared loss gradients, averaged over samples when enabled | Empirical gradient-energy proxy. Batch mode squares a batch-mean gradient; neither mode is the model-distribution Fisher information. |
| `naswot` | Log determinant of the sum of binary activation-agreement kernels | Uses ReLU-like hooks when available, capped at `naswot_max_rows` examples. |
| `jacobian` | Log of a sampled output-to-input Jacobian squared Frobenius norm per input dimension | Hutchinson estimate with random probes and output subsampling, not Jacobian covariance. |
| `conditioning` | Mean log condition number of sampled weight matrices | Does not measure the end-to-end network Jacobian. |
| `sensitivity` | Norm of the input gradient of mean squared output | The finite-difference fallback is a different directional proxy. |
| `activation_diversity` | One minus mean absolute cosine similarity between sample activations | ReLU-like layers only. |

Failures should be omitted from raw results and weighted scoring. A returned
zero is a valid score only when its `Result.success` flag is true. Hard metric
timeouts currently use daemon threads; a timed-out computation can continue to
touch the model. Candidate process isolation is needed before relying on hard
timeouts for robust concurrent GPU search.
