# Example showing how to fit a Royle-Nichols model with pyunmarked
import numpy as np
import pandas as pd
from scipy import special
from pyunmarked import UnmarkedFrame, RoyleNicholsModel

# Sample sizes
N = 300
J = 5

# truth
beta_occ = np.array([0, 0.6])
elev = np.random.randn(N).reshape(N,1)
mm_occ = np.concatenate([np.repeat(1, N).reshape(N,1), elev], axis=1)
lam = np.exp(np.matmul(mm_occ, beta_occ))

# simulate true state
z = np.random.poisson(lam, N)
zrep = np.tile(z, (J,1)).transpose()

# simulate detection probability
beta_p = np.array([-0.3, -0.4])
wind = np.random.randn(N*J).reshape(N*J, 1)
mm_det = np.concatenate([np.repeat(1,N*J).reshape(N*J,1), wind], axis=1)
r = special.expit(np.matmul(mm_det, beta_p)).reshape(N, J)
q = 1 - r
p = 1 - q**zrep

# simulate detection process
y = np.empty((N,J))
for i in range(N):
    for j in range(J):
        y[i,j] = np.random.binomial(1, p[i,j], 1)

# Construct unmarked frame
sc = pd.DataFrame({"elev": elev.reshape(N,)})
oc = pd.DataFrame({"wind": wind.reshape(N*J,)})
umf = UnmarkedFrame(y, sc, oc)

# Set up model
mod = RoyleNicholsModel("~wind","~elev", umf) 

# Fit model
mod.fit()

# Look at output
mod.summary()

# Simulate dataset and fit a model to it
ynew = mod.simulate()
umf2 = UnmarkedFrame(ynew, sc, oc)
mod2 = RoyleNicholsModel("~wind", "~elev", umf2)
mod2.fit()
mod2.summary()
