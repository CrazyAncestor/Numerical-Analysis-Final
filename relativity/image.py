import numpy as np
import matplotlib.pyplot as plt

dim = 3


#physical constant
G = 1.
M = 1.
c = 1e0
rs = 2*G*M/(c**2)

Tau = 2000.
N = 1000
dtau = Tau/N


def solve_path(start_dir,dt):
    def scalar_N(pos):
        
        r = (np.sum(pos**2))**0.5
        return ((1+rs/r)/(1-rs/r))**0.5
    def gradient_N(pos):
        
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
    
    for t in range(N):
        x1 = move(x,x,dt*0.5)
        x2 = move(x,x1,dt*0.5)
        x3 = move(x,x2,dt)
        x = (move(x,x,dt) + 2*move(x,x1,dt) + 2*move(x,x2,dt) +move(x,x3,dt) )/6.
        if(fall(x[:dim])):
            break
        
    return np.array(x)

def square(n,l):
    m = int(n/4)
    t = np.linspace(0,1.,m)
    
    x = -l/2.+l*t[:int(n/4)]
    y = np.zeros(m)+l/2.

    x = np.concatenate((x,np.zeros(m)+l/2.))
    y = np.concatenate((y,l/2.-l*t[:int(n/4)]))

    x = np.concatenate((x,l/2.-l*t[:int(n/4)]))
    y = np.concatenate((y,np.zeros(m)-l/2.))

    x = np.concatenate((x,np.zeros(m)-l/2.))
    y = np.concatenate((y,l*t[:int(n/4)]-l/2.))
    return x,y

if __name__ == '__main__':
    L = rs*100
    r0 = rs*1000
    n_image =200
    layer = 4
    n = layer*n_image
    xo ,yo = [],[]
    for t in range(layer):
        x, y = square(200,L*(t+1)/layer)
        xo = np.concatenate((xo,x))
        yo = np.concatenate((yo,y))
    zo = np.zeros(n)-r0
    
    vxo = np.zeros(n)
    vyo = np.zeros(n)
    vzo = np.zeros(n) +1.

    pos_ori = np.transpose(np.array([xo,yo,zo,vxo,vyo,vzo]))
    pos_final = []
    for t in range(n):
        pos_final.append(solve_path(pos_ori[t],dtau))

    #pos_final = pos_ori
    pos_final =np.array(pos_final)
    xf = pos_final[:,0]
    yf = pos_final[:,1]

    for i in range(layer):
        plt.plot(xo[n_image*i:n_image*(i+1)],yo[n_image*i:n_image*(i+1)], color ='blue', linewidth =1, linestyle ='-')
        plt.plot(xf[n_image*i:n_image*(i+1)],yf[n_image*i:n_image*(i+1)], color ='red', linewidth =1, linestyle ='-')
    #plt.legend(loc='lower right')
    plt.show()