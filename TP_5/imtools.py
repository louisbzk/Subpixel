
"""
imtools toolbox
author: Lionel Moisan

v0.1 (13/10/2022): initial version (View, fftzoom)
v0.2 (20/10/2022): added perdecomp, fshift, randphase, normsat
v0.3 (27/10/2022): added fsym2, ffttrans, fftshear
v0.4 (17/11/2022): added fzoom, frot (and other functions related to spline interpolation)

available functions/classes:

u = load(filename): load image
View(image): interactive visualization
v = fftzoom(image, zoom): image zooming (Shannon interpolation)
p,s = perdecomp(u): periodic + smooth decomposition
v = randphase(u): phase randomization
v = normsat(u, saturation): image renormalization (0<=saturation<=100)
v = fsym2(u): symmetrize an image along each coordinate
v = ffttrans(u, tx, ty): image translation using Shannon interpolation
v = fftshear(u, a, v, axis): shear transformation of an image using Shannon interpolation
v = fzoom(u, z, order): image zoom using spline or bicubic Keys interpolation
v = frot(u, theta, order, x0, y0): image roration using spline or bicubic Keys interpolation
"""

from matplotlib import image
import tkinter as tk
import numpy as np
from math import pi

def load(filename):
    return image.imread(filename)

class View():
    """ 
    interactive image visualizer
    zoom in: left mouse button / z
    zoom out: right mouse button / u 
    reset zoom and contrast: middle mouse button / r
    maximize contrast: c
    quit: q / escape
    """
    def __init__(self, image, name="Image Viewer"):
        self.win = tk.Tk()
        self.name = name  # nom de la fenêtre
        self.ny, self.nx = image.shape # taille de la fenêtre
        self.u = image    # image source
        self.dyn = [np.min(self.u),np.max(self.u)]
        self.x0, self.y0 = 0, 0  # coordonnées dans u du pixel haut,gauche affiché
        self.zoom = 1  # facteur de zoom
        self.zoom_method = 'nearest neighbor'  # méthode de zoom
        self.dragflag = False  # pas de glisser-déposer en cours
        self.win.title(self.name)
        self.can = tk.Canvas(self.win, height=self.ny, width=self.nx)
        self.can.pack()
        self.compute_display()
#        self.can.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.win.bind("c", self.max_contrast)
        self.win.bind("q", self.quit)        
        self.win.bind("r", self.zoom_reset)     
        self.win.bind("u", self.zoom_out)
        self.win.bind("z", self.zoom_in)        
        self.win.bind("<Escape>", self.quit)        
        self.win.bind("<Motion>", self.on_move)
        self.win.bind("<Button-1>", self.zoom_in)
        self.win.bind("<Button-2>", self.zoom_reset)        
        self.win.bind("<Button-3>", self.zoom_out)        
        self.win.mainloop()

    def compute_display(self):
        v = self.crop(self.u, self.x0, self.y0, self.nx, self.ny, self.zoom, self.zoom_method, 0)
        den = max(1e-100,self.dyn[1]-self.dyn[0])
        v = np.minimum(255,np.maximum(0,255*(v-self.dyn[0])/den))
        header = 'P5 '+str(self.nx)+' '+str(self.ny)+' 255 '
        xdata = bytes(header, 'ascii') + v.astype(dtype=np.uint8).tostring()
        self.photo = tk.PhotoImage(width=self.nx, height=self.ny, data=xdata, format='PPM')
        self.can.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
    def crop(self, u, x0, y0, nx, ny, zoom, zoom_method, bg):
        v = bg * np.ones((ny,nx))
        x = x0 + np.arange(nx)/zoom # positions sur u
        y = y0 + np.arange(ny)/zoom
        # interpolation plus proche voisin
        x = np.round(x).astype('int')
        y = np.round(y).astype('int')
        ix1 = np.argmax(x>=0)
        iy1 = np.argmax(y>=0)        
        ix2 = nx - np.argmax(x[::-1]<u.shape[1])
        iy2 = ny - np.argmax(y[::-1]<u.shape[0])        
#        v[ix1:ix2,iy1:iy2] = u[np.ix_(x[ix1:ix2],y[iy1:iy2])]
        v[iy1:iy2,ix1:ix2] = u[np.ix_(y[iy1:iy2],x[ix1:ix2])]        
        return v

    """    
    def mousewheel_callback(fig,event,im):
        redraw = false;
        if event=='WindowScrollWheel':
            dz = -event.VerticalScrollCount;
            if dz>0 or fig.UserData.z>1:
                oldz = fig.UserData.z;
                fig.UserData.z = max(1,fig.UserData.z+dz);
                if fig.UserData.z==1:
                    fig.UserData.x0 = 1;
                    fig.UserData.y0 = 1;
                else:
                    pos = mouse_location(fig);
                    # nx0 vérifie x0+xpos/oldz = nx0+xpos/newz
                    fig.UserData.x0 = fig.UserData.x0+pos(1)/oldz-pos(1)/fig.UserData.z;
                    fig.UserData.y0 = fig.UserData.y0+pos(2)/oldz-pos(2)/fig.UserData.z;
                update_figname(fig);
                redraw = true;
        else:
            printf("unrecognized event: %s\n",event.EventName);
        if redraw:
            im.CData = compute_display(fig.UserData);
    """

    # zoom in
    def zoom_in(self, event):
        oldz = self.zoom
        self.zoom *= 2
        self.x0 = self.x0 + (event.x-1)*(1/oldz - 1/self.zoom)
        self.y0 = self.y0 + (event.y-1)*(1/oldz - 1/self.zoom)
#        print("new zoom is {}".format(self.zoom))
        self.compute_display()
        self.on_move(event)
            
    # zoom reset
    def zoom_reset(self, event):
        if self.zoom!=1:
            self.zoom = 1
            self.x0, self.y0 = 0, 0
            self.compute_display()
            self.on_move(event)

    # zoom out
    def zoom_out(self, event):
        if self.zoom==2:
            self.zoom_reset(event)
        if self.zoom>1:
            oldz = self.zoom
            self.zoom //= 2
            self.x0 = self.x0 + (event.x-1)*(1/oldz - 1/self.zoom)
            self.y0 = self.y0 + (event.y-1)*(1/oldz - 1/self.zoom)
#            print("new zoom is {}".format(self.zoom))
            self.compute_display()
            self.on_move(event)            

    # maximize contrast
    def max_contrast(self, event):
        v = self.crop(self.u, self.x0, self.y0, self.nx, self.ny, self.zoom, self.zoom_method, 0)
        self.dyn = [np.min(v), np.max(v)]
        self.on_move(event)
        self.compute_display()
        
    def on_move(self, event):
        x = self.x0 + (event.x-1)/self.zoom
        y = self.y0 + (event.y-1)/self.zoom
        if self.zoom==1:
            self.win.title(self.name+"  x={:g} y={:g}  (range=[{:g},{:g}])".format(x,y,self.dyn[0],self.dyn[1]))
        else:
            self.win.title(self.name+"  x={:g} y={:g}  (range=[{:g},{:g}], zoom={})".format(x,y,self.dyn[0],self.dyn[1],self.zoom))
            
    def quit(self, event):
        self.win.destroy()

# Zoom / Unzoom of an image with Fourier interpolation
# (zero-padding / frequency cutoff)
def fftzoom(u, z=2):
    ny,nx = u.shape
    mx = int(z*nx)
    my = int(z*ny)
    dx = nx//2 - mx//2
    dy = ny//2 - my//2
    if z>=1:
        #===== zoom in
        v = np.zeros((my,mx), dtype=np.complex)
        v[-dy:-dy+ny,-dx:-dx+nx] = np.fft.fftshift(np.fft.fft2(u))
    else:
        #===== zoom out
        f = np.fft.fftshift(np.fft.fft2(u));
        v = f[dy:dy+my, dx:dx+mx]
        if mx%2==0:
            v[:, 0] = 0  # cancel non-Shannon frequencies
        if my%2==0:
            v[0, :] = 0  # cancel non-Shannon frequencies
    return z*z*np.real(np.fft.ifft2(np.fft.ifftshift(v)))

def perdecomp(u):
    """ 
    compute the periodic (p) and smooth (s) components of an image (numpy 2D array)
    note: this function also works for 1D arrays
    """
    ny, nx = u.shape
    u = u.astype('double')
    X = np.arange(nx)
    Y = np.arange(ny)
    v = np.zeros((ny, nx))
    v[ 0,X] =  u[0,X] - u[-1,X]
    v[-1,X] = -v[0,X]
    v[Y, 0] = v[Y, 0] + u[Y,0] - u[Y,-1]
    v[Y,-1] = v[Y,-1] - u[Y,0] + u[Y,-1]
    fx = np.tile(np.cos(2.*pi*X/nx), (ny,1))
    fy = np.tile(np.cos(2.*pi*Y/ny), (nx,1)).T
    fx[0,0] = 0  # avoid division by 0 in the line below
    s = np.real(np.fft.ifft2(np.fft.fft2(v)*0.5/(2.-fx-fy)))
    p = u-s
    return p, s
  
def fshift(u, dx=0, dy=0):
    "periodic shift of an image (or 1D signal)"
    if len(u.shape)==1:
        return np.roll(u, dx)
    return np.roll(np.roll(u, dy, axis=0), dx, axis=1)

def randphase(u):
    "randomize the phase of the Fourier Transform of an image"
    ny,nx = u.shape
    f = np.fft.fft2(np.random.randn(ny, nx))
    f[0,0] = 1 # preserve average value of input
    f[f==0] = 1
    f = f/np.abs(f)
    return np.real(np.fft.ifft2(np.fft.fft2(u)*f))

def normsat(u, saturation=0):
    """
    normalize intensities values of u into [0,1], allowing saturation of some pixels
    contrast is maximized given the percentage (saturation) of saturated pixels
    """
    r = np.sort(np.ndarray.flatten(u))
    n = r.shape[0]
    p = int(np.floor(saturation*0.01*n))
    if p>0:
        v = r[-p:]-r[:p]
        i = np.argmin(v)
        m = r[i]
        d = r[i+n-p-1]-m
    else:
        m = np.min(u)
        d = np.max(u) - m
    if d==0.:
        v = 0.5*np.ones(u.shape)
    else:
        v = (u-m)/d
        v[v>1.] = 1.
        v[v<0.] = 0.
    return v

def fsym2(u):
    """ 
    symmetrize an image along each coordinate (size is doubled on each axis)
    """
    v = np.vstack((u,u[::-1,:]))
    return np.hstack((v,v[:,::-1]))

def ffttrans(u, tx=0., ty=0.):
   """
   Subpixel signal/image translation using Fourier (Shannon) interpolation 
   (tx,ty) is the position in output image corresponding to (0,0) in input image
   in other terms, input value (0,0) is translated to (tx,ty)
   output image is defined by v(l,k) = U(k-tx,l-ty), where U is the Shannon interpolate of u
   """
   if u.ndim==1:
       nx, = u.shape
       u = u.astype('double')
       mx = nx//2
       P = np.arange(mx,mx+nx)%nx - mx
       Tx = np.exp(-2.*1j*pi*tx*P/nx)
       v = np.real(np.fft.ifft(np.fft.fft(u)*Tx))
   elif u.ndim==2:
       ny, nx = u.shape
       u = u.astype('double')
       mx, my = nx//2, ny//2
       P = np.arange(mx,mx+nx)%nx - mx
       Q = np.arange(my,my+ny)%ny - my
       Tx = np.tile(np.exp(-2.*1j*pi*tx*P/nx), (ny,1))
       Ty = np.tile(np.exp(-2.*1j*pi*ty*Q/ny), (nx,1)).T
       v = np.real(np.fft.ifft2(np.fft.fft2(u)*Tx*Ty))
   else:
        raise NameError("Unrecognized signal dimension (ndim should be 1 or 2)")
   return v

def fftshear(u, a, b, axis=1):
    """
    Apply an horizontal or vertical shear to an image with Fourier interpolation
    axis (0 or 1) specifies the coordinate along which the variable translation is applied
    If axis=1, output image v is defined by
      v[y, x] = U(y, x+a(y-b))    x=0..nx-1, y=0..ny-1
    where U is the Fourier interpolate of u
    """
    ny,nx = u.shape
    v = u.astype('double')
    if axis==1:
        for y in range(ny):
            v[y,:] = ffttrans(v[y,:], a*(y-b))
    elif axis==0:
        for x in range(nx):
            v[:,x] = ffttrans(v[:,x], a*(x-b))
    else:
        raise NameError("Unrecognized axis value (should be 0 or 1)")
    return v
              

# coefficients of piecewise polynomial bicubic Keys function
def mat_coeff_keys(a):
    return np.array([[0,0,a,-a], [0,-a,2*a+3,-(a+2)], [1,0,-a-3,a+2], [0,a,-2*a,a]])

from math import factorial, comb

# coefficients of piecewise polynomial spline function with order n
def mat_coeff_splinen(n):
    c = np.zeros((n+1,n+1))
    a = [0]*(n+2)
    a[0] = 1/factorial(n)
    for k in range(1, n+2):
        a[k] = -a[k-1]*(n+2-k)/k
    for k in range(n+2):
        for p in range(n+1):
            xp = comb(n,p)*k**(n-p)
            for i in range(k, n+1):
                c[i,p] += a[i-k]*xp
    return c

# 1D inverse z-Transform (on all lines) with poles z
def invspline1D(c, z):
    npoles = len(z)
    N = c.shape[1]
    # normalization
    c *= np.prod([(1-x)*(1-1/x) for x in z])
    for k in range(npoles): # loop on poles
        # initialize causal filter, symmetric boundary conditions
        c[:,0] = np.hstack((1., z[k]**np.arange(1,N-1) + z[k]**np.arange(2*N-3,N-1,-1), z[k]**(N-1))) @ c.T / (1-z[k]**(2*N-1))
        for n in range(1,N): #forward recursion 
            c[:,n] += z[k]*c[:,n-1]
        # initialize anticausal filter, symmetric boundary conditions
        c[:,N-1] = (z[k]/(z[k]**2-1)) * (z[k]*c[:,N-2] + c[:,N-1])
        for n in range(N-2,-1,-1): # backwards recursion 
            c[:,n] = z[k]*(c[:,n+1]-c[:,n]);
    return c

# 2D inverse B-spline transform
def finvspline(c,order):
    # initialize poles of associated z-filter 
    if order==2:
        z = [-0.17157288]  # sqrt(8)-3 
    elif order==3: 
        z = [-0.26794919]  # sqrt(3)-2 
    elif order==4: 
        z = [-0.361341,-0.0137254]
    elif order==5:
        z = [-0.430575,-0.0430963]
    elif order==6:
        z = [-0.488295,-0.0816793,-0.00141415]
    elif order==7:
        z = [-0.53528,-0.122555,-0.00914869]
    elif order==8:
        z = [-0.574687,-0.163035,-0.0236323,-0.000153821]
    elif order==9:
        z = [-0.607997,-0.201751,-0.0432226,-0.00212131]
    elif order==10: 
        z = [-0.636551,-0.238183,-0.065727,-0.00752819,-0.0000169828]
    elif order==11:
        z = [-0.661266,-0.27218,-0.0897596,-0.0166696,-0.000510558]
    else:
        raise NameError('finvspline: order should be in 2..11.')
    return invspline1D(invspline1D(c, z).T, z).T


def fzoom(u, z=2, order=0):
    """
    Zoom an image by a factor z 

    z is not necessarily an integer (default value: 2)
    available interpolations are:
       order = 0           nearest neighbor (default)
       order = 1           bilinear
       order = -3          bicubic Keys
       order = 3,5,7,9,11  spline with specified order
    """
    u = u.astype('double')
    ny, nx = u.shape
    sx = int(nx*z)
    sy = int(ny*z)
    if order==0:
        # nearest neighbor is a special case (symmetric interpolation grid)
        X = (nx-1-(sx-1)/z)/2 + np.arange(sx)/z
        Y = (ny-1-(sy-1)/z)/2 + np.arange(sy)/z
        iX = sx - np.floor(sx-X+0.5).astype(int)
        iY = sy - np.floor(sy-Y+0.5).astype(int)
        v = u[np.ix_(iY,iX)]
    else:
        X = np.arange(sx)/z;
        Y = np.arange(sy)/z
        iX = np.floor(X).astype(int)
        iY = np.floor(Y).astype(int)
        X -= iX
        Y -= iY
        if order==-3: # bicubic Keys interpolation
            n2 = 2
            c = mat_coeff_keys(-1/2)
            order = 3
        elif order in {1,3,5,7,9,11}: # spline interpolation
            n2 = (order+1)//2
            c = mat_coeff_splinen(order)
            if order>1: 
                u = finvspline(u, order)
        else:
            raise NameError('fzoom: Unrecognized interpolation order.')
        n1 = 1-n2
        # compute interpolation coefficients
        cx = c @ ( X**np.vstack(np.arange(order+1)) )
        cy = c @ ( Y**np.vstack(np.arange(order+1)) )        
        # add (symmetrical) borders to avoid undefined coordinates
        u = np.hstack((u[:, -n1:0:-1], u, u[:, -2:-n2-2:-1]))
        u = np.vstack((u[-n1:0:-1, :], u, u[-2:-n2-2:-1, :]))    
        # interpolation
        v = np.zeros((sy,sx))
        for dx in range(n1,n2+1):
            for dy in range(n1,n2+1):
                v += np.vstack(cy[n2-dy,:]) * cx[n2-dx,:] * u[np.ix_(iY+dy-n1-1,iX+dx-n1-1)]
    return v

def frot(u, theta, order=1, x0=None, y0=None):
    """ 
    Rotate an image 
    
    The output image v is defined by
    v(y,x) = U(Y,X)  where U is the specified interpolate of u and
    
    X = x0 + cos(theta)*(x-x0) - sin(theta)*(y-y0)
    Y = y0 + sin(theta)*(x-x0) + cos(theta)*(y-y0)
    
    by default (if x0 and y0 are not both specified) the rotation center is
    the image center: (x0,y0) = ( (nx+1)/2, (ny+1)/2 )
    
    The rotation angle (theta) is counted counterclockwise, in degrees
    
    available interpolations are:
     order = 0           nearest neighbor (default)
     order = 1           bilinear
     order = -3          bicubic Keys
     order = 3,5,7,9,11  spline with specified order
    """
    u = u.astype('double')
    ny, nx = u.shape
    if x0 is None:
        x0 = (nx-1)/2
    if y0 is None:
        y0 = (ny-1)/2
    x, y = np.meshgrid(range(nx), range(ny))
    co = np.cos(theta*np.pi/180)
    si = np.sin(theta*np.pi/180)    
    X = x0 + co*(x-x0) - si*(y-y0)
    Y = y0 + si*(x-x0) + co*(y-y0)
    if order==0: # nearest neighbor interpolation
        iX = np.floor(X+0.5).astype(int)
        iY = np.floor(Y+0.5).astype(int)
        n1, n2 = 0, 0
    else: # other interpolations
        iX = np.floor(X).astype(int)
        iY = np.floor(Y).astype(int)
        X -= iX
        Y -= iY
        if order==-3: # bicubic Keys interpolation
            n2 = 2
            c = mat_coeff_keys(-1/2)
            order = 3
        elif order in {1,3,5,7,9,11}: # spline interpolation
            n2 = (order+1)//2
            c = mat_coeff_splinen(order)
            if order>1:
                u = finvspline(u, order) 
        else:
            raise NameError("Unrecognized interpolation order.")
        n1 = 1-n2
    # add (symmetrical) borders to avoid undefined coordinates
    iX = iX.flatten()
    iY = iY.flatten()
    supx1 = max(0, -min(iX)) - n1
    supx2 = max(0, max(iX)-nx+1) + n2
    u = np.hstack((u[:,supx1:0:-1], u, u[:,nx-2:nx-2-supx2:-1]))
    iX = iX + supx1
    supy1 = max(0, -min(iY)) - n1
    supy2 = max(0, max(iY)-ny+1) + n2
    u = np.vstack((u[supy1:0:-1,:], u, u[ny-2:ny-2-supy2:-1,:]))
    iY = iY + supy1
    if order==0: # nearest neighbor interpolation
        v = np.reshape(u.flatten()[iX+iY*u.shape[1]], (ny,nx))
    else: # other interpolations 
        # compute interpolation coefficients
        cx = c @ ( X.flatten()**np.vstack(np.arange(order+1)) )
        cy = c @ ( Y.flatten()**np.vstack(np.arange(order+1)) )        
        # interpolation
        v = np.zeros(nx*ny)
        for dx in range(n1, n2+1):
            for dy in range(n1, n2+1):
                v += cy[n2-dy,:] * cx[n2-dx,:] * u.flatten()[iX+dx-n1-1+(iY+dy-n1-1)*u.shape[1]]
        v = np.reshape(v, (ny,nx))
    return v

