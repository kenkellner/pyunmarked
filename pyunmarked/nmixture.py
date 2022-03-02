from . import model
import numpy as np
from scipy import special, stats

class NmixtureModel(model.UnmarkedModel):
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
        p = mod["det"].predict(beta=beta_det, interval=False).reshape(N, J)
        nll = 0.0
        for i in range(N):       
            kvals = range(int(mod.response.Kmin[i]), int(K)+1)
            f = stats.poisson.pmf(kvals, lam[i])
            ymat = np.tile(y[i,], (len(kvals), 1)) 
            pmat = np.tile(p[i,], (len(kvals), 1))
            kmat = np.tile(kvals, (J, 1)).transpose()
            g = stats.binom.logpmf(ymat, kmat, pmat).sum(axis=1)
            fg = f * np.exp(g)
            nll -= np.log(fg.sum())
        
        return nll
