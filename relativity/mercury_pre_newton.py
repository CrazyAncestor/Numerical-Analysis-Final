import numpy as np
import matplotlib.pyplot as plt

# Physical constants
m = 1. #mercury mass
M = 1.989e30 #sun mass

G = 6.67e-11# gravitational constant

c = 299792458  #light speed#You can touch

#L_peri = 6.98179e10 
#V_peri = 58976.*4.6/6.98179
L_peri = 4.60012e10 
V_peri = 58976.
L = m*V_peri*L_peri

E = 0.5*(V_peri**2) -G*M/L_peri
a = -G*M/(2*E)

alpha = 3*(L**2)/(c**2)/(m**2)*1e0#Adjustment Parameter#You can touch

P = 100 #Number of rotation#You can touch

N = 1000*P # number of time intervals
T = 2*np.pi*(a**1.5)/((G*M)**0.5)*P # total time elapse

dt = T/N # time interval
Period = 2*np.pi*(a**1.5)/((G*M)**0.5)#period of one circle

def force_add(x,v):
    fc = -G*M*m*x/((np.sum(x**2))**1.5)
    fr = -G*M*m*x/((np.sum(x**2))**2.5)*alpha
    return fc+fr

def move_newton(x,v,x1,v1,deltat):
    xn = x + v1*deltat
    vn = v + force_add(x1,v1)/m*deltat
    return np.array([xn,vn])

def solve_rk4_newton(start_pos,start_vel):
    
   x0 =start_pos
   x=x0
   v0 =start_vel
   v=v0

   track = [x0]

   for t in range(N):
       x1,v1 = move_newton(x,v,x,v,dt*0.5)
       x2,v2 = move_newton(x,v,x1,v1,dt*0.5)
       x3,v3 = move_newton(x,v,x2,v2,dt)
       f = (move_newton(x,v,x,v,dt) + 2*move_newton(x,v,x1,v1,dt) + 2*move_newton(x,v,x2,v2,dt) +move_newton(x,v,x3,v3,dt) )/6.
       x,v = f[0],f[1]
       track.append(x)
   
   return np.array(track)
## TODO:
# def solve_rk5_newton():

# def perihelion(track):
#    dist = np.sqrt(track[:,0]**2 + track[:,1]**2)
#    p = np.where(dist == np.amin(dist))
#    return p[0]

# def apogee(track):
#    dist = np.sqrt(track[:,0]**2 + track[:,1]**2)
#    a = np.where(dist == np.amax(dist))
#    return a[0]

def garbage(track):
   perihelion = []
   apogee = []
   def dist(p):
      return np.sqrt(p[0]**2+p[1]**2)

   _save = True
   for i in range(track.shape[0]):
      _prev = i-1
      _next = i+1
      if i == 0:
         _prev = -1
      if i == track.shape[0]-1:
         _next = 0
      if dist(track[i]) < dist(track[_prev]) and dist(track[i]) < dist(track[_next]):
         if _save:
            perihelion.append(track[i])
         #_save = not _save
      if dist(track[i]) > dist(track[_prev]) and dist(track[i]) > dist(track[_next]):
         if _save:
            apogee.append(track[i])
         #_save = not _save
   #perihelion = perihelion[:len(perihelion)-1]
   apogee = apogee[:len(apogee)-1]

   return np.array(perihelion), np.array(apogee)




if __name__ == '__main__':
    start = np.array([L_peri,0])
    v0 = np.array([0,V_peri])
    t  = np.linspace(0,T,N+1)
    
    option ='newton'
    if option=='newton':

        rk4_newton = solve_rk4_newton(start,v0)
        x_newton = rk4_newton[:,0]
        y_newton = rk4_newton[:,1]

    #     init_angle = np.arctan((rk4_newton[0]-rk4_newton[half])[1]/(rk4_newton[0]-rk4_newton[half])[0])
    #     last_angle = np.arctan((rk4_newton[-1]-rk4_newton[-1-half])[1]/(rk4_newton[-1]-rk4_newton[-1-half])[0])
    #     print('precession angle:'+str(last_angle-init_angle))
        
        #plt.plot(x_newton,y_newton, color ='green', linewidth =1, linestyle ='-',label = 'newton')
        plt.plot(x_newton,y_newton, color ='green', label = 'newton')

        p, a = garbage(rk4_newton)
        print("perihelion: ",p)
        print("apogee: ",a)
        plt.plot(p[:,0], p[:,1], color ='red', label = 'perihelion')
        plt.plot(a[:,0], a[:,1], color ='blue', label = 'apogee')
        
        # first = p[0]-a[0]
        # last = p[-1]-a[-1]
        # rad = np.arctan(last[1]/last[0]) - np.arctan(first[1]/first[0])
        rad = np.arctan(p[-1][1]/p[-1][0]) - np.arctan(p[0][1]/p[0][0])
        print("arc second:", rad/2/np.pi*360*3600)
        rad = np.arctan(a[-1][1]/a[-1][0]) - np.arctan(a[0][1]/a[0][0])
        print("arc second:", rad/2/np.pi*360*3600)
    
    plt.legend(loc='lower left')
    plt.show()
        
    
