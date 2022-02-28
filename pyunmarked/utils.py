import numpy as np

# Inverse logit transformation
def inv_logit(lp):
    return np.exp(lp) / (1 + np.exp(lp))
