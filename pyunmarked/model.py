import numpy as np
import pandas as pd
from scipy import stats, optimize
import patsy
import prettytable


class Response(object):
    def __init__(self, y):
        self.y = y
        self.Kmin = np.apply_along_axis(max, 1, y)


class Submodel(object):    
    def __init__(self, name, code, formula, invlink, data):
        self.name = name
        self.code = code
        self.formula = formula
        self.invlink = invlink
        self.data = data
        self.estimates = None
        self.vcov = None
        self.coefnames = self.get_coefnames()
    
    def dmatrix(self):
        return patsy.dmatrix(self.formula, self.data)
    
    def get_coefnames(self):
        return self.dmatrix().design_info.column_names
    
    def npars(self):
        return len(self.coefnames)
    
    def predict(self, transform=True, interval=True, level=95, beta=None):
        level = level/100 if level > 1 else level
        beta = self.estimates if beta is None else beta
        if beta is None:
            raise AttributeError("Model has not been fit yet")
        modmat = self.dmatrix()
        lp = np.matmul(modmat, beta)
        if not interval:
            if not transform:
                return lp
            else:
                return self.invlink(lp)
        
        vcov = np.matmul(np.matmul(modmat, self.vcov), modmat.transpose())
        se = np.sqrt(vcov.diagonal())
        ci = stats.norm.interval(level, loc=lp, scale=se)
        out = pd.DataFrame({"Prediction":lp, "lower":ci[0], "upper":ci[1]})
        
        if transform:
            out = self.invlink(out)
        return out
    
    def get_vcov(self, opt):
        self.vcov = opt.hess_inv[self.index,:][:,self.index]
    
    def get_estimates(self, opt):
        self.estimates = opt.x[self.index]
     
    def check_fit(self):
        if self.estimates is None:
            raise AttributeError("Model has not been fit yet")
    
    def SE(self):
        self.check_fit()
        return np.sqrt(self.vcov.diagonal())
    
    def confint(self, level=95):
        level = level/100 if level > 1 else level
        return stats.norm.interval(level, loc=self.estimates, scale=self.SE())
    
    def summary(self, level=95):
        level = level/100 if level > 1 else level
        ci = self.confint(level)
        tab = prettytable.PrettyTable()
        tab.add_column("Parameter", self.coefnames)
        tab.add_column("Estimate", self.estimates.round(4))
        tab.add_column("SE", self.SE().round(4))
        tab.add_column("lower", ci[0].round(4))
        tab.add_column("upper", ci[1].round(4))
        print(self.name+": "+self.formula)
        print(tab)
    
    def print(self, level=95):
        self.summary(level=level)
    
    def coeftable(self, level=95):
        ci = self.confint(level)
        return pd.DataFrame({"Model": np.repeat(self.code, len(self.coefnames)),
            "Parameter": self.coefnames, "Estimate": self.estimates,
            "SE": self.SE(), "lower": ci[0], "upper": ci[1]})


class SubmodelDict(object):
    def __init__(self, **args):
        self.submodels = args
        idx = 0
        for i in self.submodels:
            self.submodels[i].index = np.arange(len(self.submodels[i].coefnames))
            self.submodels[i].index += idx
            idx += len(self.submodels[i].index)
    
    def npars(self):
        return sum(dict.values({key: x.npars() for key, x in self.submodels.items()}))
    
    def get_estimates(self, opt):
        for i in self.submodels:
            self.submodels[i].get_estimates(opt)
    
    def get_vcov(self, opt):
        for i in self.submodels:
            self.submodels[i].get_vcov(opt)
    
    def summary(self, level=95):
        for i in self.submodels:
            self.submodels[i].summary(level=level)
            print("")
    
    def coeftable(self, level=95):
        tabs = {k: x.coeftable(level=level) for k, x in mod.submodels.submodels.items()}
        tabs = pd.concat(tabs)
        return tabs.reset_index(drop=True)
     
    def print(self, level=95):
        self.summary(level=level)



class UnmarkedModel(object):
    def __init__():
        pass
                         
    def __getitem__(self, arg):
        return self.submodels.submodels[arg]
    
    def negloglik(self, x, mod):
        pass
        
    def fit(self, x0=None, gtol=None):
        gtol = 1e-6 * self.response.y.shape[0] if gtol is None else gtol
        #gtol = 1e-6 if gtol is None else gtol
        x0 = np.repeat(0, self.submodels.npars()) if x0 is None else x0
        self.opt = optimize.minimize(self.negloglik, x0, self, method="BFGS",
                options={"gtol": gtol})
        self.submodels.get_estimates(self.opt)
        self.submodels.get_vcov(self.opt)
         
    def check_fit(self):
        if 'opt' not in dir(self):
            raise AttributeError("Model has not been fit yet")
         
    def AIC(self):
        self.check_fit()
        return 2 * self.opt.fun + 2 * self.submodels.npars()
        
    def summary(self, level=95):
        self.submodels.summary(level=level)
        print("AIC: "+str(round(self.AIC(), 4)))
        print("Converged: "+str(self.opt.success))
    
    def coeftable(self, level=95):
        return self.submodels.coeftable(level=level)
    
    def predict(self, type, transform=True, interval=True, level=95):
        return self[type].predict(transform=transform, interval=interval,
                level=level)
    
    def simulate(self):
        pass
