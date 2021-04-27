import numpy as np
import matplotlib.pyplot as plt
import re

# IAU Convention - Resolution B3
# http://arxiv.org/abs/1510.07674
GmEarth_m3s2 = 3.986004e14 # m^3/s2
GmJupiter_m3s2 = 1.2668653e17 # m^3/s2
GmSun_m3s2 = 1.3271244e20 # m^3/s2
mEarth_Sun = GmEarth_m3s2/GmSun_m3s2

# Speed of light
c_m_s = 299792458

# Draw a color map
def _colormap_calc_edges(x, y):
  if x.ndim == 1:
    xg = x
  else:
    xg = x[0]
  if y.ndim == 1:
    yg = y
  else:
    yg = y[:,0]
  dx = xg[1:]-xg[:-1]
  dy = yg[1:]-yg[:-1]
  xg = np.concatenate(([xg[0]-dx[0]/2],xg[1:]-dx/2,[xg[-1]+dx[-1]/2]))
  yg = np.concatenate(([yg[0]-dy[0]/2],yg[1:]-dy/2,[yg[-1]+dy[-1]/2]))
  xedges, yedges = np.meshgrid(xg, yg)
  return(xedges, yedges)

def colormap(x, y, z, clabel="color label", vmin=None, vmax=None):
  """
  Plot the colormap of z in the x,y plane.
  clabel is the label of the colorbar (label for z).
  vmin, vmax are the min/max values for the colorbar.
  Values of z below vmin are plotted with the color corresponding to vmin,
  and values above vmax are plotted with the color of vmax (saturation).
  """
  xedges, yedges = _colormap_calc_edges(x, y)
  znan = np.ma.array(z, mask=np.isnan(z))
  plt.pcolormesh(xedges, yedges, znan,
    vmin=vmin, vmax=vmax, cmap='Blues_r', rasterized=True)
  plt.axis([xedges.min(), xedges.max(), yedges.min(), yedges.max()])
  cbar = plt.colorbar()
  cbar.set_label(clabel)

def M2E(M, e, ftol=1e-14, Nmax=50):
  """
  Compute eccentric anomaly from mean anomaly (and eccentricity).
  """
  E = M.copy()
  deltaE = np.array([1])
  N = 0
  while max(abs(deltaE))>ftol and N<Nmax:
    diff = M-(E-e*np.sin(E))
    deriv = 1-e*np.cos(E)
    deltaE = diff/deriv
    E += deltaE
    N += 1
  return(E)

def E2v(E, e):
  """
  Compute true anomaly from eccentric anomaly (and eccentricity).
  """
  v = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
  return(v)

def v2E(v, e):
  """
  Compute eccentric anomaly from true anomaly (and eccentricity).
  """
  E = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(v/2))
  return(E)

def E2M(E, e):
  """
  Compute mean anomaly from eccentric anomaly (and eccentricity).
  """
  M = E - e*np.sin(E)
  return(M)

def star_rv(phase, K_m_s, e, omega_rad):
  """
  Compute the radial velocity of the star in the center of mass rest frame,
  at given phases (array) and with given values of the orbital parameters
  (K, e, omega).
  """
  E0_rad = v2E(np.array([np.pi/2-omega_rad]), e)
  M0_rad = E2M(E0_rad, e)[0]
  M_rad = M0_rad + phase*2*np.pi
  E_rad = M2E(M_rad, e)
  th_rad = E2v(E_rad, e)
  rv_m_s = K_m_s*(np.cos(th_rad+omega_rad) + e*np.cos(omega_rad))
  return(rv_m_s)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
