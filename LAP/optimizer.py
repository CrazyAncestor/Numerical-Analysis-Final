import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


L = 5.  # total length of the route

M = 100 # number of time intervals
T = 1.  # total time elapse
dt = T/M # time interval

# Physical constants
m=1. # mass
g = 10. # gravitational accleration

# Main function to solve the track
def solve_lagrangian(start_pos,end_pos):
    # All the position points(including head and tail)
        #linearly distributed initialization
    track = np.linspace(start_pos,end_pos,M)
    
    # Track points(head and tail,excluded)
    track = np.delete(track,0,0)
    
    def total_s(track):# Total action
        # All the position points(including head and tail)
        pos = np.append(track,end_pos)
        pos = np.insert(pos,0,start_pos)
        
        def delta_s (init_pos,final_pos):#action within a time interval

            def Lagrangian(m,v,pos):
                K = 0.5* m *(v**2)
                V = -m * g * pos
                return K - V

            pos_ave =( init_pos + final_pos)/2.
            v = (final_pos - init_pos)/dt
            return dt * Lagrangian(m,v,pos_ave)
        
        return np.sum(delta_s(pos[0:-1],pos[1:]))    
    
    result = opt.minimize(total_s,track)
    track = np.array(result.x)
    track = np.append(track,end_pos)
    track = np.insert(track,0,start_pos)
    return track

if __name__ == '__main__':
    start = L *0
    end   = L *1.0
    t  = np.linspace(0,T,M+1)
    track = solve_lagrangian(start,end)
    stnd = 5. *(t**2)
    
    
    plt.plot(t, track, color ='blue', linewidth =2, linestyle ='-',label='exact solution')
    plt.plot(t, stnd, color ='red', linewidth =2, linestyle ='--',label='solve by scipy.optimize')
    plt.legend(loc='upper left')
    plt.show()
    
