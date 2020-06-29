import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


L = 1.  # total length of the rout

N = 100 # number of time intervals
T = float(np.pi*8)  # total time elapse
dt = T/N # time interval

# Physical constants
m = 1. # mass
k = 1. #elastic constant
b = 1. #damping factor
g = 10.# gravitational accleration

def force(x,v):
    return -k*x
def move(x,v,x1,v1,deltat):
    xn = x + v1*deltat
    vn = v + force(x1,v1)/m*deltat
    return np.array([xn,vn])
def solve_euler_newton(start_pos,start_vel):
    
   x0 =start_pos
   x=x0
   v0 =start_vel
   v=v0

   track = [x0]
   dt = T/N

   for t in range(N):
       f = move(x,v,x,v,dt)
       x,v = f[0],f[1]
       track.append(x)
    
   return np.array(track)

def solve_rk2_newton(start_pos,start_vel):
    
   x0 =start_pos
   x=x0
   v0 =start_vel
   v=v0

   track = [x0]
   dt = T/N

   for t in range(N):
       x1,v1 = move(x,v,x,v,dt*0.5)
       f = move(x,v,x1,v1,dt)
       x,v = f[0],f[1]
       track.append(x)
    
   return np.array(track)

def solve_rk4_newton(start_pos,start_vel):
    
   x0 =start_pos
   x=x0
   v0 =start_vel
   v=v0

   track = [x0]
   dt = T/N

   for t in range(N):
       x1,v1 = move(x,v,x,v,dt*0.5)
       x2,v2 = move(x,v,x1,v1,dt*0.5)
       x3,v3 = move(x,v,x2,v2,dt)
       f = (move(x,v,x,v,dt) + 2*move(x,v,x1,v1,dt) + 2*move(x,v,x2,v2,dt) +move(x,v,x3,v3,dt) )/6.
       x,v = f[0],f[1]
       track.append(x)
    
   return np.array(track)

if __name__ == '__main__':
    start = L *0
    end   = L *1.0
    v0 = 1.
    t  = np.linspace(0,T,N+1)

    euler_newton = solve_euler_newton(start,v0)
    rk2_newton = solve_rk2_newton(start,v0)
    rk4_newton = solve_rk4_newton(start,v0)
    stnd = np.sin(t)
    print("rk2 error:"+str(np.sum((rk2_newton-stnd)**2)/len(stnd)))
    print("rk4 error:"+str(np.sum((rk4_newton-stnd)**2)/len(stnd)))
    
    plt.plot(t, stnd, color ='blue', linewidth =2, linestyle ='-',label = 'exact solution')
    plt.plot(t, euler_newton, color ='green', linewidth =2, linestyle =':',label = 'euler_newton')
    plt.plot(t, rk2_newton, color ='brown', linewidth =2, linestyle =':',label = 'rk2_newton')
    plt.plot(t, rk4_newton, color ='red', linewidth =2, linestyle ='--',label = 'rk4_newton')
    plt.legend(loc='upper left')
    plt.show()
