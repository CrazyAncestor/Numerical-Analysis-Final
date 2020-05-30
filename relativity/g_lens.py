import numpy as np
import matplotlib.pyplot as plt

dim = 2


#physical constant
G = 1.
M = 1.
c = 1e0
rs = 2*G*M/(c**2)

Tau = 100.
N = int(Tau*1e2)
dtau = Tau/N


def solve_path(start_dir,dt):
    def scalar_N(pos):
        #x = pos[0]
        #y = pos[1]
        r = (np.sum(pos**2))**0.5
        return ((1+rs/r)/(1-rs/r))**0.5
    def gradient_N(pos):
        x = pos[0]
        y = pos[1]
        r = (np.sum(pos**2))**0.5
        return -2*rs/((r-rs)**2)*pos/r
    def move(u,r,deltat):
        #ne'=gradient_n - e(gradient_n.e)
        pos = r[:dim]
        e = r[dim:]
        diff_pos = e
        diff_e = (gradient_N(pos)-e*np.dot(gradient_N(pos),e))/scalar_N(pos)
        diff = np.concatenate((diff_pos,diff_e),axis=None)
        return u+diff*deltat
    def fall(pos):
        r = np.sum(np.array(pos)**2)
        if r<((rs*1.01)**2):
            return True
    x0 =start_dir
    x = x0
    track = [x0]
    for t in range(N):
        x1 = move(x,x,dt*0.5)
        x2 = move(x,x1,dt*0.5)
        x3 = move(x,x2,dt)
        x = (move(x,x,dt) + 2*move(x,x1,dt) + 2*move(x,x2,dt) +move(x,x3,dt) )/6.
        if(fall(x[:dim])):
            break
        track.append(x)
    return np.array(track)

if __name__ == '__main__':
    theta = np.array([0.1,0.2,0.3])
    color = ['blue','red','green']

    for i in range(1):
        init_pos=np.array([-10,0,np.cos(np.pi/10000*i+np.pi*3.38/10),np.sin(np.pi/10000*i+np.pi*3.38/10)])
        path = solve_path(init_pos,dtau)
        print(len(path))
        tau = np.linspace(0,Tau,len(path))
        x = path[:,0]
        y = path[:,1]

        plt.plot(x,y, color ='blue', linewidth =1, linestyle ='-')#,label = ('pi/20*'+str(int(i)))

    theta = np.linspace(0,2*np.pi,1000)
    x_hole = rs*np.cos(theta)
    y_hole = rs*np.sin(theta)
    x_hole1 = rs*1.5*np.cos(theta)
    y_hole1 = rs*1.5*np.sin(theta)
    plt.plot(x_hole,y_hole, color ='black', linewidth =1, linestyle ='-',label = 'black hole')
    plt.plot(x_hole1,y_hole1, color ='yellow', linewidth =1, linestyle ='-',label = 'light hole')
    plt.legend(loc='lower right')
    plt.show()
