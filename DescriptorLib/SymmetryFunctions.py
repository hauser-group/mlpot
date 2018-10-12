import numpy as _np
import warnings
    
class CosCutoffFunction(object):
    def __init__(self, cut):
        self.cut = cut
    
    def __call__(self, r):
        return (r <= self.cut)*0.5*(_np.cos(_np.pi*r/self.cut)+1)
        
    def derivative(self, r):
        return (r <= self.cut)*(-0.5*_np.pi*_np.sin(_np.pi*r/self.cut)/self.cut)
        
        
class TanhCutoffFunction(object):
    def __init__(self, cut):
        self.cut = cut
        
    def __call__(self, r):
        return (r <= self.cut)*_np.tanh(1-r/self.cut)**3       
        
    def derivative(self, r):
        return (r <= self.cut)*(-3.0*_np.tanh(1-r/self.cut)**2 /
                (_np.cosh(1-r/self.cut)**2*self.cut))
    
    
class RadialSymmetryFunction(object):
    def __init__(self, rs, eta, cut, cutoff_type = "cos"):
        self.rs = rs
        self.eta = eta
        if cutoff_type == "cos":
            self.cut_fun = CosCutoffFunction(cut)
        elif cutoff_type == "tanh":
            self.cut_fun = TanhCutoffFunction(cut)
        else:
            warnings.warn("{} not recognized, switching to 'cos'".format(cutoff_type),
                          UserWarning)
            self.cut_fun = CosCutoffFunction(cut)
        
    def __call__(self, r):
        return _np.exp(-self.eta*(r-self.rs)**2)*self.cut_fun(r)
        
    def derivative(self, r):
        return (-2.0*self.eta*(r-self.rs)*self(r) +
                _np.exp(-self.eta*(r-self.rs)**2)*self.cut_fun.derivative(r))

class AngularSymmetryFunction(object):
    def __init__(self, eta, zeta, lamb, cut, cutoff_type = "cos"):
        self.eta = eta
        self.zeta = zeta
        self.lamb = lamb
        if cutoff_type == "cos":
            self.cut_fun = CosCutoffFunction(cut)
        elif cutoff_type == "tanh":
            self.cut_fun = TanhCutoffFunction(cut)
        else:
            warnings.warn("{} not recognized, switching to 'cos'".format(cutoff_type),
                          UserWarning)
            self.cut_fun = CosCutoffFunction(cut)
        
    def __call__(self, rij, rik, costheta):
        return 2**(1-self.zeta)* ((1 + self.lamb*costheta)**self.zeta * 
            _np.exp(-self.eta*(rij**2+rik**2)) * self.cut_fun(rij)*self.cut_fun(rik))
            
    def derivative(self, rij, rik, costheta):
        sintheta = _np.sqrt(1-costheta**2)
        return 2**(1-self.zeta)*(-self.lamb * self.zeta * sintheta *
            (1 + self.lamb*costheta)**(self.zeta-1) * 
            _np.exp(-self.eta*(rij**2+rik**2)) * self.cut_fun(rij)*self.cut_fun(rik))
        
class AngularSymmetryFunctionNew(object):
    def __init__(self, eta, zeta, lamb, rs, cut, cutoff_type = "cos"):
        self.eta = eta
        self.zeta = zeta
        self.lamb = lamb
        if cutoff_type == "cos":
            self.cut_fun = CosCutoffFunction(cut)
        elif cutoff_type == "tanh":
            self.cut_fun = TanhCutoffFunction(cut)
        else:
            warnings.warn("{} not recognized, switching to 'cos'".format(cutoff_type),
                          UserWarning)
            self.cut_fun = CosCutoffFunction(cut)
        
    def __call__(self, rij, rik, costheta):
        return 2**(1-self.zeta)* ((1 + self.lamb*costheta)**self.zeta * 
            _np.exp(-self.eta*((rij-self.rs)**2+(rik-self.rs)**2)) * self.cut_fun(rij)*self.cut_fun(rik))
            
    def derivative(self, rij, rik, costheta):
        sintheta = _np.sqrt(1-costheta**2)
        return 2**(1-self.zeta)*(-self.lamb * self.zeta * sintheta *
            (1 + self.lamb*costheta)**(self.zeta-1) * 
            _np.exp(-self.eta*((rij-self.rs)**2+(rik-self.rs)**2)) * self.cut_fun(rij)*self.cut_fun(rik))

def radial_function(rij, rs, eta, cut):
    return _np.exp(-eta*(rij-rs)**2)*cutoff_function(rij,cut)
    
def angular_function(rij, rik, costheta, eta, zeta, lamb,  cut):
    return 2**(1-zeta)* ((1 + lamb*costheta) * _np.exp(-eta*(rij**2+rik**2)) *
        cutoff_function(rij,cut)*cutoff_function(rik,cut))
        
def cutoff_function(r,cut):
    # Weird formulation allows for calls using vectors
    return (r <= cut)*0.5*(_np.cos(_np.pi*r/cut)+1)