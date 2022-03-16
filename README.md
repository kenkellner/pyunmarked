# pyunmarked

A python port of the R package [unmarked](https://cran.r-project.org/web/packages/unmarked/index.html), for analyzing ecological data while accounting for imperfect detection.

### Fit an occupancy model

```python
import numpy as np
import pandas as pd
from scipy.special import expit
from pyunmarked import OccupancyModel, UnmarkedFrame

# Simulate an occupancy dataset
N = 1000 # Number of sites
J = 5 # Number of sampling occasions

# Occupancy probability parameters
beta_occ = np.array([0, 0.5])
# Occupancy covariate
elev = np.random.randn(N).reshape(N,1)
# Model matrix
mm_occ = np.concatenate([np.repeat(1, N).reshape(N,1), elev], axis=1)
# Occupancy probability
psi = expit(np.matmul(mm_occ, beta_occ))

# Simulate true latent state
z = np.random.binomial(1, psi, N)

# Detection probability parameters
beta_p = np.array([0, -0.4])
# Detection covariate
wind = np.random.randn(N*J).reshape(N*J, 1)
# Model matrix
mm_det = np.concatenate([np.repeat(1,N*J).reshape(N*J,1), wind], axis=1)
# Detection probability
p = expit(np.matmul(mm_det, beta_p)).reshape(N, J)

# Simulate detection process
y = np.empty((N,J)) # presence-absence data matrix
for i in range(N):
    for j in range(J):
        y[i,j] = np.random.binomial(1, z[i]*p[i,j], 1)

# Organize covariates into DataFrames
sc = pd.DataFrame({"elev": elev.reshape(N,)})
oc = pd.DataFrame({"wind": wind.reshape(N*J,)})

# Create UnmarkedFrame containing y and covariates
umf = UnmarkedFrame(y, sc, oc)

# Fit occupancy model
mod = OccupancyModel("~wind", "~elev", umf)
mod.fit()

# Look at summary of output
mod.summary()

# Calculate predicted occupancy at each site
mod.predict("occ")
```
