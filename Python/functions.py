import numpy as np
from scipy import linalg, signal
from scipy.interpolate import interp1d, interp2d, interpn
from scipy.sparse import coo_matrix
from scipy.signal import medfilt2d, convolve2d, convolve
from scipy.sparse.linalg import LinearOperator, cg
import time as timer
import pdb

def optApod(Nelem, Nlags, tol=np.finfo(float).eps):
    '''OPTAPOD Find apodization that maximizes short lag autocorrelation
    apod = optApod(Nelem,Nlags,tol,print)
        Nelem = number of elements in aperture
        Nlags = number of lags in optimization
        tol = (optional) numerical tolerance for optimization (default: tol = eps)'''
    
    # Laplacian with Homogeneous BCs
    LagNDiff = np.diag(2*Nlags*np.ones(Nelem));
    for lag in np.arange(1,Nlags+1):
        LagNDiff = LagNDiff - \
            np.diag(np.ones(Nelem-lag),k=lag) - \
            np.diag(np.ones(Nelem-lag),k=-lag);
    
    # Solve Ball-Exclusion Constrained Optimization Problem
    apod = np.random.randn(Nelem); # Initial Apodization
    prev_lambda = 0; curr_lambda = 1; # Lagrange Multiplier
    while (np.abs(np.linalg.norm(apod)-1) > tol) and (np.abs((curr_lambda-prev_lambda)/curr_lambda) > tol):
    	prev_lambda = curr_lambda; # Update Previous Lagrange Multiplier
    	apod = np.abs(apod)/np.linalg.norm(apod); # Project Onto Nonnegative Norm-2 Ball
    	LagNDiffConstrained = np.vstack((np.hstack((LagNDiff, apod[:,np.newaxis])), np.hstack((apod, 0))));
    	sol = np.linalg.solve(LagNDiffConstrained, np.hstack((np.zeros(Nelem), 1)));
    	apod = sol[:Nelem]; # Extract Optimal Apodization
    	curr_lambda = sol[-1] # New Lagrange Multiplier
    return apod	
	

def propagate(x, z1, z2, c, f, rxdata_z1_f, txdata_z1_f, aawin):
    '''rxdata_z2_f, txdata_z2_f = propagate(x, z1, z2, c, f, rxdata_z1_f, txdata_z1_f, aawin)
    
    PROPAGATE - Angular Spectrum Propagation of TX/RX Signals into the Medium
    This function propagates transmit and receive wavefields at from one
    depth to another depth using the angular spectrum method
    
    INPUTS:
    x                  - 1 x X vector of x-grid positions for wavefield
    z1                 - depth of input TX and RX wavefields
    z2                 - depth of output TX and RX wavefields
    c                  - 1 x X vector of speed of sound [m/s] between z1 and z2
    f                  - 1 x F vector of pulse frequencies in spectrum
    rxdata_z1_f        - X x F x N array of input RX wavefields at z1
    txdata_z1_f        - X x F x N array of input TX wavefields at z1
    aawin              - 1 x X vector of lateral taper to prevent wraparound
    
    OUTPUT:
    rxdata_z2_f        - X x F x N array of output RX wavefields at z2
    txdata_z2_f        - X x F x N array of output TX wavefields at z2'''

    # Verify the Number of Common Shot Gathers
    ns = txdata_z1_f.shape[2]; assert(rxdata_z1_f.shape[2] == ns), \
        'Number of sources must equal to number of common-source gathers';
    AAwin = np.tile(aawin[:,np.newaxis,np.newaxis].astype(np.single), (1, f.size, ns));

    # Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
    ft = lambda sig: np.fft.fftshift(np.fft.fft(AAwin*sig, axis=0).astype(np.csingle), axes=0);
    ift = lambda sig: AAwin*np.fft.ifft(np.fft.ifftshift(sig, axes=0), axis=0).astype(np.csingle);

    # Spatial Grid
    dx = np.mean(np.diff(x)); nx = x.size;

    # FFT Axis for Lateral Spatial Frequency
    kx = np.mod(np.fft.fftshift(np.arange(nx,dtype=np.single)/(dx*nx))+1/(2*dx), 1/dx)-1/(2*dx);
    
    # Mean Slowness and Lateral Spatial Frequency
    s = 1/c.astype(np.single) # Slowness [s/m]
    smean = np.mean(s) # Mean Slowness [s/m]
    ds = s - smean # Lateral Variation in Slowness [s/m] Along x
    
    # Continuous Wave Response By Downward Angular Spectrum
    F, Kx = np.meshgrid(f.astype(np.single),kx); # Create Grid in f-kx
    _, Ds = np.meshgrid(f.astype(np.single),ds); # Create Grid
    Kz = np.sqrt(((F*smean)**2-Kx**2).astype(np.csingle)); # Axial Spatial Frequency
    H = np.exp(1j*2*np.pi*Kz*(z2-z1)); # Propagation Filter
    H[Kz**2 <= 0] = 0; # Remove Evanescent Components
    H = np.tile(H[:,:,np.newaxis], (1,1,ns)); # Replicate Across Shots
    dH = np.exp(1j*2*np.pi*F*Ds*(z2-z1)); # Phase Screen
    dH = np.tile(dH[:,:,np.newaxis], (1,1,ns)); # Replicate Across Shots
    
    # Apply Propagation Filter
    rxdata_z2_f = dH*ift(H*ft(rxdata_z1_f));
    txdata_z2_f = np.conj(dH)*ift(np.conj(H)*ft(txdata_z1_f));
    return rxdata_z2_f, txdata_z2_f;
    

def line_pixel_intersection( x_grid, y_grid, start_pt, end_pt ):
    '''LINE_PIXEL_INTERSECTION Calculate Intersection of Line on Grid Pixels
    USAGE:
    	[ grid, intersegments ] = line_pixel_intersection( x_grid, y_grid, start_pt, end_pt )
    INPUTS:
        x_grid, y_grid -- Dict Specifying X,Y-Coordinate of Grid  
                          ['start']: start position
                          ['spacing']: position spacing on grid
                          ['N']: number of points on grid
        start_pt       -- Start Point of Line [x, y]
        end_pt         -- End Point of Line [x, y]
    OUTPUTS:
        grid           -- Dict with (['x'], ['y']) grid on which intersegments calculated 
        intersegments  -- Dict Specifying Intersection of Line with Pixels
                          ['fragEndPts']: (x, y) coords of intersegment endpoints
                          ['lengths']: dict of line-pixel intersection lengths in
                              (i, j, val) format where (i, j) is the row and column
                              in matrix and val is the length over the pixel.
                              specified by ['row'], ['col'], ['val']. '''

    # Create Grid for Pixel Center
    grid = {}; 
    grid['x'] = x_grid['start'] + np.arange(x_grid['N']) * x_grid['spacing']; 
    grid['y'] = y_grid['start'] + np.arange(y_grid['N']) * y_grid['spacing'];
     
    # Create Grid for Pixel Vertices 
    xv = x_grid['start'] - x_grid['spacing']/2 + np.arange(x_grid['N']+1) * x_grid['spacing']; 
    yv = y_grid['start'] - y_grid['spacing']/2 + np.arange(y_grid['N']+1) * y_grid['spacing']; 
    
    # Find Line-Pixel Intersections
    xrng = np.array([start_pt[0], end_pt[0]]);
    yrng = np.array([start_pt[1], end_pt[1]]);
    if xrng[0] != xrng[1]:
    	line_param_x = (xv-xrng[0])/(xrng[1]-xrng[0]);
    else:
    	line_param_x = np.array([]);
    if yrng[0] != yrng[1]:
    	line_param_y = (yv-yrng[0])/(yrng[1]-yrng[0]);
    else:
    	line_param_y = np.array([]);
    line_param = np.sort(np.hstack((line_param_x, line_param_y)));
    line_param = np.hstack((0, line_param[np.logical_and(line_param>0, line_param<1)], 1))
    xint = xrng[0] + line_param * (xrng[1]-xrng[0]);
    yint = yrng[0] + line_param * (yrng[1]-yrng[0]);
    
    # Sparse Matrix of Intersection Lengths
    intersegments = {}
    intersegments['lengths'] = {}
    intersegments['lengths']['row'] = \
        np.round(((yint[:-1]+yint[1:])/2-y_grid['start'])/y_grid['spacing'])+1;
    intersegments['lengths']['col'] = \
        np.round(((xint[:-1]+xint[1:])/2-x_grid['start'])/x_grid['spacing'])+1;
    intersegments['lengths']['val'] = np.sqrt(np.diff(xint)**2 + np.diff(yint)**2);
    intersegments['fragEndPts'] = np.hstack((xint[:,np.newaxis],yint[:,np.newaxis]));
    return grid, intersegments;


def spray(val_mask, mask):
    '''SPRAY Adjoint of Masking Operation Used to Sample Values
    val = spray(val_mask, mask) 
    INPUT:
    	val_mask = values at masked locations
    	mask = masked locations in output array [logical array]
    	    (numel(val_mask) must equal nnz(mask))
    OUTPUT:
        val = full array with values at masked locations and zero elsewhere'''
    val = np.zeros(mask.shape);
    val[mask] = val_mask;
    return val;


# Define Loadmat Function for HDF5 Format ('-v7.3' in MATLAB)
import h5py
def loadmat_hdf5(filename):
    file = h5py.File(filename,'r')
    out_dict = {}
    for key in file.keys():
        out_dict[key] = np.ndarray.transpose(np.array(file[key]));
    file.close()
    return out_dict;

# Python-Equivalent Command for IMAGESC in MATLAB
import matplotlib.pyplot as plt
def imagesc(x, y, img, rng, cmap='gray', numticks=(3, 3), aspect='equal'):
    exts = (np.min(x)-np.mean(np.diff(x)), np.max(x)+np.mean(np.diff(x)), \
        np.min(y)-np.mean(np.diff(y)), np.max(y)+np.mean(np.diff(y)));
    plt.imshow(np.flipud(img), cmap=cmap, extent=exts, vmin=rng[0], vmax=rng[1], aspect=aspect);
    plt.xticks(np.linspace(np.min(x), np.max(x), numticks[0]));
    plt.yticks(np.linspace(np.min(y), np.max(y), numticks[1]));
    plt.gca().invert_yaxis();

# MATLAB-Equivalent Functions
gausswin = lambda N, alpha: signal.windows.gaussian(N, std=(N-1)/(2*alpha));
db = lambda value: 20*np.log10(value);
