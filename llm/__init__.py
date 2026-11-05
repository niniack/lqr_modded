"""LQR activation steering with linear block models.

model         load the model, record the state entering each block
concepts      sentence CSV, concept direction E
jacobian      mean_jacobian and opt-in efficient_mean_jacobian (chunked jacrev)
dmd           clean-state DMD of each block
steering      Riccati gains, LQRSteering, nominal-reference DriftLQRSteering, sampling
trace         steered vs unsteered replay of sampled text, metrics, model errors
steering_dmd  DMD fitted on steering deviations
judge         Granite judge and a repetition score
"""
