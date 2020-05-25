import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

dim = 4

m =1.
M = 1.989e30 #sun mass
G = 6.67e-11# gravitational constant
c = 299792458  #light speed
rs = 2*G*M/(c**2)#Schwarzschild radius
gmc2 = G*M/(c**2)
gm = G*M

r_peri = 4.6001200e10 #6.98e10##e10 
v_peri = 5.8976e04#*4.6/6.98#58976.
gamma = (1-(v_peri/c)**2)**(-0.5)
P = 100
n = 1e03
N = n*P # number of time intervals
L = m*r_peri*v_peri
E = 0.5*(v_peri**2) -G*M/r_peri
a = -G*M/(2*E)
Period = 2*np.pi*(a**1.5)/((G*M)**0.5) #*gamma
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
            track.append(x)
            
        return np.array(track)

def garbage(track):
   perihelion = []
   apogee = []
   
   _save = True
   for i in range(track.shape[0]):
      _prev = i-1
      _next = i+1
      if i == 0:
         _prev = -1
      if i == track.shape[0]-1:
         _next = 0
      if track[i] < track[_prev] and track[i] < track[_next]:
         if _save:
            perihelion.append(i)
         #_save = not _save
      if track[i] > track[_prev] and track[i] > track[_next]:
         if _save:
            apogee.append(i)
         #_save = not _save
   #perihelion = perihelion[:len(perihelion)-1]
   apogee = apogee[:len(apogee)-1]
   perihelion = perihelion[:len(perihelion)-1]

   return np.array(perihelion), np.array(apogee)   

if __name__ == '__main__':
    init_pos = np.array([0,r_peri,float(np.pi/2.),0.  ,gamma,0.,0.,v_peri/r_peri*gamma])
    
    mercury = planet(init_pos)
    sol =mercury.solve_rk4()
    t = sol[:,0]
    r = sol[:,1]
    psi = sol[:,3]
    x = r*np.cos(psi)
    y = r*np.sin(psi)


    p, a = garbage(r)
    print("perihelion: ",p)
    print("apogee: ",a)
    #plt.plot(p[:,0], p[:,1], color ='red', label = 'perihelion')
    #plt.plot(a[:,0], a[:,1], color ='blue', label = 'apogee')
        
    # first = p[0]-a[0]
    # last = p[-1]-a[-1]
    # rad = np.arctan(last[1]/last[0]) - np.arctan(first[1]/first[0])
    rad = psi[p[-1]] - psi[p[0]] -2*np.pi*(p[-1]- p[0])/n#np.arctan(p[-1][1]/p[-1][0]) - np.arctan(p[0][1]/p[0][0])
    print("arc second:", rad/2/np.pi*360*3600)
    rad = psi[a[-1]] - psi[a[0]] -2*np.pi*(a[-1]- a[0])/n
    print("arc second:", rad/2/np.pi*360*3600)



    plt.plot(x,y, color ='green', linewidth =1, linestyle ='-',label = 'track')
    
    plt.show()
