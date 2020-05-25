import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

dim = 4
N = 1000

m =1.
M = 1.#e30 #sun mass
G = 1.# 6.67e-11# gravitational constant
c = 1.#299792458  #light speed
rs = 2*G*M/(c**2)#Schwarzschild radius
gmc2 = G*M/(c**2)
gm = G*M

r_fac = 1.3
v_fac = 0.9
r_peri = rs*r_fac#e10 
v_peri = (gm/r_peri)**0.5*v_fac#58976.
gamma = (1-(v_peri/c)**2)**(-0.5)

P = 0.33
N = 1e04*P # number of time intervals
T = 2*np.pi*r_peri*P/v_peri/gamma# total time elapse

dt = T/N

class connection(object):
    def __init__(self,x):

        self.x = np.zeros([dim,dim,dim])
        t = x[0]
        r = x[1]
        theta = x[2]
        psi = x[3]

        fac = 1.-rs/r
        
        self.x[0][0][1] = gmc2/(r**2)/fac
        self.x[0][1][0] = gmc2/(r**2)/fac

        self.x[1][0][0] = gm*fac/(r**2)
        self.x[1][1][1] = -gmc2/(r**2)/fac
        self.x[1][2][2] = -r*fac
        self.x[1][3][3] = -r*fac*(np.sin(theta)**2)

        self.x[2][1][2] = 1./r
        self.x[2][2][1] = 1./r
        self.x[2][3][3] = -np.sin(theta)*np.cos(theta)

        self.x[3][1][3] = 1./r
        self.x[3][3][1] = 1./r
        self.x[3][2][3] = np.cos(theta)/np.sin(theta)
        self.x[3][3][2] = np.cos(theta)/np.sin(theta)
        
        
def f(t,r):
    x = np.zeros(dim)
    v = np.zeros(dim)
    for i in range(dim):
        x[i] = r[i]
        v[i] = r[i+dim]
    con = connection(x)
    diff_x = v
    diff_v = np.zeros(dim)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                diff_v[i] -= con.x[i][j][k]*v[j]*v[k]
    
    return np.concatenate((diff_x,diff_v),axis=None)

def move(u,r,deltat):
    x = np.zeros(dim)
    v = np.zeros(dim)
    for i in range(dim):
        x[i] = r[i]
        v[i] = r[i+dim]
    con = connection(x)
    diff_x = v
    diff_v = np.zeros(dim)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                diff_v[i] -= con.x[i][j][k]*v[j]*v[k]
    

    diff = np.concatenate((diff_x,diff_v),axis=None)
    return u+diff*deltat
    

def solve_rk4(start_pos):
    
   x0 =start_pos
   x=x0
   track = [x0]
   
   for t in range(int(N)):
       x1 = move(x,x,dt*0.5)
       x2 = move(x,x1,dt*0.5)
       x3 = move(x,x2,dt)
       x = (move(x,x,dt) + 2*move(x,x1,dt) + 2*move(x,x2,dt) +move(x,x3,dt) )/6.
       track.append(x)
    
   return np.array(track)

if __name__ == '__main__':
    init_pos = np.array([0,r_peri,float(np.pi/2.),0.  ,gamma,0.,0.,v_peri/r_peri*gamma])
    
    deltat = T
    sol =solve_rk4(init_pos)

    print(sol[0])
    print(sol[-1])

    t = sol[:,0]
    r = sol[:,1]
    psi = sol[:,3]
    x = r*np.cos(psi)
    y = r*np.sin(psi)


    plt.plot(x,y, color ='green', linewidth =1, linestyle ='-',label = 'track')
    
    plt.show()
