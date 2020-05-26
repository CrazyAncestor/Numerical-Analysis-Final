import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

dim = 2
N = 1000
class connection(object):
    def __init__(self,x):

        self.x = np.zeros([dim,dim,dim])
        theta = x[0]
        psi = x[1] 
        
        self.x[0][0][0] = 0

        self.x[0][1][0] = 0
        self.x[0][0][1] = 0

        self.x[0][1][1] = -np.sin(theta)*np.cos(theta)

        self.x[1][0][0] = 0

        self.x[1][1][0] = np.cos(theta)/np.sin(theta)
        self.x[1][0][1] = np.cos(theta)/np.sin(theta)
        
        self.x[1][1][1] = 0

def f(t,r):
    x = np.zeros(dim)
    v = np.zeros(dim)
    for i in range(dim):
        x[i] = r[i]
        v[i] = r[i+dim]
    con = connection(x)
    diff_x = v
    diff_v = -np.tensordot(np.tensordot(con.x,v,axes=([2],[0])),v,axes=([1],[0]))
    
    return np.concatenate((diff_x,diff_v),axis=None)

if __name__ == '__main__':
    init_pos = np.array([float(np.pi/2.),0.,0.,1.])
    deltat = np.pi
    sol =solve_ivp(f,(0,deltat),init_pos,t_eval=np.linspace(0,deltat,N+1),method='DOP853')
    print(sol.y[:,-1])
