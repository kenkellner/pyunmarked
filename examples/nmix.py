# Example script for fitting N-mixture model with pyunmarked
import numpy as np
import pandas as pd
from scipy.special import expit
from pyunmarked import UnmarkedFrame, NmixtureModel

N = 300
J = 5

# truth
beta_occ = np.array([0, 0.6])
elev = np.random.randn(N).reshape(N,1)
mm_occ = np.concatenate([np.repeat(1, N).reshape(N,1), elev], axis=1)
lam = np.exp(np.matmul(mm_occ, beta_occ))

# simulate true state
z = np.random.poisson(lam, N)

# simulate detection probability
beta_p = np.array([0, -0.4])
wind = np.random.randn(N*J).reshape(N*J, 1)
mm_det = np.concatenate([np.repeat(1,N*J).reshape(N*J,1), wind], axis=1)
p = expit(np.matmul(mm_det, beta_p)).reshape(N, J)

# simulate detection process
y = np.empty((N,J))
for i in range(N):
    for j in range(J):
        y[i,j] = np.random.binomial(z[i], p[i,j], 1)


# Construct unmarked frame
sc = pd.DataFrame({"elev": elev.reshape(N,)})
oc = pd.DataFrame({"wind": wind.reshape(N*J,)})
umf = UnmarkedFrame(y, sc, oc)

# Set up model
nmix_mod = NmixtureModel("~wind", "~elev", umf) 

# Fit model
nmix_mod.fit()

# Look at results
nmix_mod.summary()
nmix_mod.coeftable()

# Generate predicted abundances
nmix_mod.predict("abun")

# Simulate a new dataset and fit a model to it
y2 = nmix_mod.simulate()
umf2 = UnmarkedFrame(y2, sc, oc)
mod2 = NmixtureModel("~wind","~elev", umf2)
mod2.fit()
mod2.summary()
