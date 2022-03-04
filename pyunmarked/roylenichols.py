from . import model
import numpy as np
from scipy import special, stats

class RoyleNicholsModel(model.UnmarkedModel):
    def __init__(self, det_formula, abun_formula,  data):
        self.response = model.Response(data.y)
        abun = model.Submodel("Abundance", "abun", abun_formula, np.exp, data.site_covs)
        det = model.Submodel("Detection", "det", det_formula, special.expit, data.obs_covs)
        self.submodels = model.SubmodelDict(abun=abun, det=det)
    
    def negloglik(self, x, mod, K):
        x = np.array(x)
        beta_abun = x[mod["abun"].index]
        beta_det = x[mod["det"].index]
        y = mod.response.y
        N, J = y.shape
        lam = mod["abun"].predict(beta=beta_abun, interval=False)
        r = mod["det"].predict(beta=beta_det, interval=False).reshape(N, J)
        q = 1 - r
        nll = 0.0
        for i in range(N):
            kvals = range(int(mod.response.Kmin[i]), int(K)+1)
            f = stats.poisson.pmf(kvals, lam[i])
            ymat = np.tile(y[i,], (len(kvals), 1)) 
            qmat = np.tile(q[i,], (len(kvals), 1))
            kmat = np.tile(kvals, (J, 1)).transpose()
            pmat = 1 - qmat**kmat
            g = stats.binom.logpmf(ymat, 1, pmat).sum(axis=1)
            fg = f * np.exp(g)
            nll -= np.log(fg.sum())
        
        return nll
    
    def simulate(self):
        N, J = self.response.y.shape
        lam = self.predict("abun", interval=False)
        q = 1 - self.predict("det", interval=False).reshape(N, J)
        z = np.random.poisson(lam, N)
        zrep = np.tile(z, (J,1)).transpose()
        p = 1 - q**zrep
        y = np.empty((N, J))
        for i in range(N):
            y[i,] = np.random.binomial(1, p[i,], J)
        return y
