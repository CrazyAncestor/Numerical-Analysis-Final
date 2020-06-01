import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

dim = 4

m =1.
M = 1. #sun mass
G = 1.# gravitational constant
c = 1.  #light speed
rs = 2*G*M/(c**2)#Schwarzschild radius
gmc2 = G*M/(c**2)
gm = G*M

r0 = rs*2.2
v0 = (G*M/r0)**0.5
gamma = (1-(v0/c)**2)**(-0.5)
P = 4.
n = 1e03
N = n*P # number of time intervals
L = m*r0*v0
E = 0.5*(v0**2) -G*M/r0
a = -G*M/(2*E)
Period = 2*np.pi*(a**1.5)/((G*M)**0.5)/gamma
T = Period*P# total time elapse

dt = T/N

class planet(object):
    def __init__(self,start_pos):
        self.start_pos = start_pos
    def connection(self,x):
        con = np.zeros([dim,dim,dim])
        t = x[0]
        r = x[1]
        theta = x[2]
        psi = x[3]

        fac = 1.-rs/r
        
        con[0][0][1] = gmc2/(r**2)/fac
        con[0][1][0] = gmc2/(r**2)/fac

        con[1][0][0] = gm*fac/(r**2)
        con[1][1][1] = -gmc2/(r**2)/fac
        con[1][2][2] = -r*fac
        con[1][3][3] = -r*fac*(np.sin(theta)**2)

        con[2][1][2] = 1./r
        con[2][2][1] = 1./r
        con[2][3][3] = -np.sin(theta)*np.cos(theta)

        con[3][1][3] = 1./r
        con[3][3][1] = 1./r
        con[3][2][3] = np.cos(theta)/np.sin(theta)
        con[3][3][2] = np.cos(theta)/np.sin(theta)
        return con
        
    def move(self,u,r,deltat):
        x = np.zeros(dim)
        v = np.zeros(dim)
        for i in range(dim):
            x[i] = r[i]
            v[i] = r[i+dim]
        con = self.connection(x)
        diff_x = v
        diff_v = -np.tensordot(np.tensordot(con,v,axes=([2],[0])),v,axes=([1],[0]))

        diff = np.concatenate((diff_x,diff_v),axis=None)
        return u+diff*deltat
        

    def solve_rk4(self):
        x0 =self.start_pos
        x=x0
        track = [x0]
        
        for t in range(int(N)):
            x1 = self.move(x,x,dt*0.5)
            x2 = self.move(x,x1,dt*0.5)
            x3 = self.move(x,x2,dt)
            x = (self.move(x,x,dt) + 2*self.move(x,x1,dt) + 2*self.move(x,x2,dt) +self.move(x,x3,dt) )/6.
            if x[1]<(1.03*rs):
                print('v/c:'+str((((x[5])**2)/(1-rs/x[1])+(x[1]*x[7])**2)**0.5/c))
                break
            track.append(x)
            
        return np.array(track)

if __name__ == '__main__':
    plt.figure(figsize=(20,10))
    ax = []
    for ep in range(2):
        v_fac = [[1.,0.9,0.99,0.999,],[1.,1.001,1.01,1.1]]#
        col = ['blue','red','green','brown']
        
        ax.append(plt.subplot(1,2,ep+1))
        for i in range(len(v_fac[ep])):
            init_pos = np.array([0,r0,float(np.pi/2.),0.  ,gamma,0.,0.,v_fac[ep][i]*v0/r0*gamma])
            
            sol = planet(init_pos)
            sol =sol.solve_rk4()
            t = sol[:,0]
            r = sol[:,1]
            psi = sol[:,3]
            x = r*np.cos(psi)
            y = r*np.sin(psi)
            
            ax[ep].plot(x,y, color =col[i], linewidth =1, linestyle ='-',label = str(v_fac[ep][i]))

        theta = np.linspace(0,2*np.pi,1000)
        x_hole = rs*np.cos(theta)
        y_hole = rs*np.sin(theta)
    
    
        ax[ep].plot(x_hole,y_hole, color ='black', linewidth =1, linestyle ='-',label = 'black hole')
        ax[ep].legend(loc='lower left')
    plt.show()
