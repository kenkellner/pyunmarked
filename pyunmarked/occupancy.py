from . import utils, model
import numpy as np

class OccupancyModel(model.UnmarkedModel):
    def __init__(self, det_formula, occ_formula,  data):
        self.response = model.Response(data.y)
        occ = model.Submodel("Occupancy", "occ", occ_formula, utils.inv_logit, data.site_covs)
        det = model.Submodel("Detection", "det", det_formula, utils.inv_logit, data.obs_covs)
        self.submodels = model.SubmodelDict(occ=occ, det=det)
    
    def negloglik(self, x, mod):
        x = np.array(x)
        beta_occ = x[mod["occ"].index]
        beta_det = x[mod["det"].index]
        y = mod.response.y
        N, J = y.shape
        psi = mod["occ"].predict(beta=beta_occ, interval=False)
        p = mod["det"].predict(beta=beta_det, interval=False).reshape(N, J)
        no_detect = 1 - mod.response.Kmin           
        nll = 0.0
        
        for i in range(N):
            cp = 1.0
            for j in range(J):
                cp *= pow(p[i,j], y[i,j]) * pow(1-p[i,j], 1-y[i,j])
            
            nll -= np.log(psi[i] * cp + (1-psi[i]) * no_detect[i])
        
        return nll
    
    def simulate(self):
        N, J = self.response.y.shape
        psi = self.predict("occ", interval=False)
        p = self.predict("det", interval=False).reshape(N, J)
        z = np.random.binomial(1, psi, N)
        y = np.empty((N, J))
        for i in range(N):
            for j in range(J):
                y[i,j] = np.random.binomial(1, z[i]*p[i,j], 1)
        return y
