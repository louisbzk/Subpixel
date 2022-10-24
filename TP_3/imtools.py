
"""
imtools toolbox
author: Lionel Moisan

v0.1 (13/10/2022): initial version (View, fftzoom)
v0.2 (20/10/2022): added perdecomp, fshift, randphase, normsat

available functions/classes:

u = load(filename): load image
View(image): interactive visualization
v = fftzoom(image, zoom): image zooming (Shannon interpolation)
p,s = perdecomp(u): periodic + smooth decomposition
v = randphase(u): phase randomization
v = normsat(u, saturation): image renormalization (0<=saturation<=100)
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

