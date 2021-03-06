import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Physical constants
m = 1. #mercury mass
M = 1.989e30 #sun mass

G = 6.67e-11# gravitational constant

c = 299792458  #light speed
e = 0.20563
L_peri = 46.001200e09  
V_peri = 58976.
L = m*V_peri*L_peri
E = 0.5*(V_peri**2) -G*M/L_peri
a = -G*M/(2*E)

P = 1000
N = 10000 # number of time intervals
T = 2*np.pi*(a**1.5)/((G*M)**0.5)*P # total time elapse
dt = T/N # time interval




def force(x,v):
    return -G*M*m*x/((np.sum(x**2))**1.5)

def move_newton(x,v,x1,v1,deltat):
    xn = x + v1*deltat
    vn = v + force(x1,v1)/m*deltat
    return np.array([xn,vn])

def move_polar(u,w,u1,w1,deltat):
    un = u + w1*deltat
    wn = w + (G*M*(m**2)/(L**2) - u1 )*deltat#+ 3*G*M*(u**2)/(c**2)
    return np.array([un,wn])

def move_relativity(u,w,u1,w1,deltat,c):
    un = u + w1*deltat
    wn = w + (G*M*(m**2)/(L**2) - u1 + 3*G*M*(u1**2)/(c**2))*deltat
    return np.array([un,wn])

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

def solve_rk4_polar(start_pos,start_vel):
    
   x0 =start_pos
   x=x0
   v0 =start_vel
   v=v0

   track = [x0]
   
   dt = 2*np.pi/N
   for t in range(N*P):
       x1,v1 = move_polar(x,v,x,v,dt*0.5)
       x2,v2 = move_polar(x,v,x1,v1,dt*0.5)
       x3,v3 = move_polar(x,v,x2,v2,dt)
       f = (move_polar(x,v,x,v,dt) + 2*move_polar(x,v,x1,v1,dt) + 2*move_polar(x,v,x2,v2,dt) +move_polar(x,v,x3,v3,dt) )/6.
       x,v = f[0],f[1]
       track.append(x)
   
   return np.array(track)

def solve_rk4_relativity(start_pos,start_vel,c):
    
   x0 =start_pos
   x=x0
   v0 =start_vel
   v=v0

   track = [x0]
   
   dt = 2*np.pi/N
   for t in range(N*P):
       x1,v1 = move_relativity(x,v,x,v,dt*0.5,c)
       x2,v2 = move_relativity(x,v,x1,v1,dt*0.5,c)
       x3,v3 = move_relativity(x,v,x2,v2,dt,c)
       f = (move_relativity(x,v,x,v,dt,c) + 2*move_relativity(x,v,x1,v1,dt,c) + 2*move_relativity(x,v,x2,v2,dt,c) +move_relativity(x,v,x3,v3,dt,c) )/6.
       x,v = f[0],f[1]
       track.append(x)
    
   return np.array(track)

if __name__ == '__main__':
    start = np.array([L_peri,0])
    v0 = np.array([0,V_peri])
    t  = np.linspace(0,T,N+1)

    u_start = 1/L_peri
    w_start = 0
    
    #x_std = np.cos(t)
    #y_std = np.sin(t)
    
    """rk4_newton = solve_rk4_newton(start,v0)
    x_newton = rk4_newton[:,0]
    y_newton = rk4_newton[:,1]"""

    theta = np.linspace(0,2*P*np.pi,N*P+1)
    rk4_polar = solve_rk4_polar(u_start,w_start)
    x_polar = (1/rk4_polar)*np.cos(theta)
    y_polar = (1/rk4_polar)*np.sin(theta)

    rk4_relativity = solve_rk4_relativity(u_start,w_start,c)
    x_relativity = (1/rk4_relativity)*np.cos(theta)
    y_relativity = (1/rk4_relativity)*np.sin(theta)
    
    polar_error = (rk4_polar[-1]-rk4_polar[0])
    relativity_error = (rk4_relativity[-1]-rk4_relativity[0])

    print('polar error:'+str(polar_error))
    print('relativity error:'+str(relativity_error))
    print('expectation:'+str(G*M*(m**2)*e/(L**2)*(np.cos(6*P*np.pi*((G*M*m/(c*L))**2))-1)))
    print('result'+str(relativity_error-polar_error))
    
    #plt.plot(x_std,y_std, color ='blue', linewidth =2, linestyle ='-',label = 'standard')
    #plt.plot(x_newton,y_newton, color ='red', linewidth =2, linestyle ='-',label = 'newton')
    plt.plot(x_polar,y_polar, color ='blue', linewidth =2, linestyle ='-',label = 'polar')
    plt.plot(x_relativity,y_relativity, color ='green', linewidth =2, linestyle =':',label = 'relativity')
    
    plt.legend(loc='lower left')
    plt.show()
        
    
