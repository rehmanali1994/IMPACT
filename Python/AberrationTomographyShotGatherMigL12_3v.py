from functions import *

# Load File and Original k-Wave Simulation Grid
load_filename = '../Datasets/RatAbdomenL12-3v.mat';
dataset = loadmat_hdf5(load_filename);
rxAptPos = dataset['rxAptPos']
time = dataset['time'][0]
full_synth_data = dataset['full_synth_data']
del dataset;
nT, nRx, nTx = full_synth_data.shape; 
tx_elmts = np.arange(nTx);
rxdata_h = full_synth_data[:,:,tx_elmts]; 
del full_synth_data;

## Define Parameters
dov = 25e-3; # Max Depth [m]
upsamp_x = 2; # Upsampling in x (assuming dx = pitch)
upsamp_z = 1; # Upsampling in z (assuming dz = lambda/2)
Nx0 = 256; # Number of Points Laterally in x
cbfm = 1540; # initial sound speed [m/s]
fTx = 7.5e6; # frequency [Hz]
fBW = 9e6; # bandwidth at half maximum [Hz]
pulse_cutoff = 1e-3; # passband cutoff
ord = 100; # Strength of Anti-Aliasing Window 
# Parameters to Filter Time Delay Measurements
Fnum = 1.0; # f-number to limit acceptance angle
dmin = 0.004; # start depth [m]
dmax = 0.020; # end depth [m]
dwnsmp = 5; # downsample measurements for tomography
medfilt_kernel = np.array([9,9]); # Size of Median Filter
reg_smoothing = 1e2; # Regularization for Direction of Arrival Smoothing
# B-Mode Image Parameters/Imaging Window
dBrange = np.array([-80, 0]); # Dynamic Range [decibels]
Nximg = 1001; Nzimg = 1001;
xlims = np.array([-20e-3, 20e-3]); 
zlims = np.array([0e-3, 25e-3]);
# Sound Speed Reconstruction Parameters
blur_z = gausswin(100, 4); # Z - Blurring Over Reconstruction Grid
blur_x = gausswin(100, 4); # X - Blurring Over Reconstruction Grid
max_iter = 20; tol = 1e-3; # Conjugate Gradient Parameters
    # NOTE max_iter is much less than in MATLAB code because 
    # MATLAB's pcg has very different behavior from scipy.sparse.linalg's cg function
    # Results from each iteration will be slightly different from MATLAB because of this
reg = 1e-2; # Regularization Corresponding to Relative Time of Flight Error
alpha = 0.5; # Relative Weighting of Noninformative Prior to Layered Prior
    # 1.0 for full noninformative prior; 0.0 for fully layered prior
num_sos_iterations = 10; # Number of Sound Speed Estimations/Iterations

## Create Simulation, Image Reconstruction, and Sound Speed Grids
lambda_ = cbfm/fTx; # wavelength [m]
pitch = np.mean(np.diff(rxAptPos[:,0])) # number of elements
no_elements = rxAptPos.shape[0] # element spacing [m]
xpos = pitch*np.arange(-(no_elements-1)/2,1+(no_elements-1)/2); # element position [m]
# Grid for Simulation
x = (pitch/upsamp_x)*np.arange(-(upsamp_x*Nx0-1)/2,1+(upsamp_x*Nx0-1)/2); # m
Nu1 = np.round(dov/((lambda_/2)/upsamp_z)); 
z = (np.arange(Nu1))*(lambda_/2)/upsamp_z; # m
xmax = (np.max(np.abs(xpos))+np.max(np.abs(x)))/2; # Lateral Cutoff for Anti-Aliasing [m]
aawin = 1/np.sqrt(1+(x/xmax)**ord); # Anti-Aliasing Window
# Grid for Image Reconstruction
x_img = np.linspace(xlims[0], xlims[1], Nximg);
z_img = np.linspace(zlims[0], zlims[1], Nzimg);
X_IMG, Z_IMG = np.meshgrid(x_img, z_img);
# Grid For Sound Speed [m/s] Estimate
dx = np.mean(np.diff(x)); Nx = int(x.size); Nz = int(np.ceil((z[-1]-z[0])/dx)+1); 
x_grid = {}; x_grid['start'] = x[0]; x_grid['spacing'] = dx; x_grid['N'] = Nx;
z_grid = {}; z_grid['start'] = z[0]; z_grid['spacing'] = dx; z_grid['N'] = Nz;
Crecon = cbfm*np.ones((Nz,Nx)); # Initial Speed of Sound [m/s]
xrecon = x_grid['start'] + x_grid['spacing']*np.arange(x_grid['N']);
zrecon = z_grid['start'] + z_grid['spacing']*np.arange(z_grid['N']);
Xrecon, Zrecon = np.meshgrid(xrecon, zrecon);

## Frequency-Domain Representation of Transmitted and Received Wavefields
# Transmit Impulse Response in Frequency Domain
nt = time.size; # [s]
fs = 1/np.mean(np.diff(time)); # [Hz] 
f = (fs/2)*np.arange(-1,1,2/nt); # [Hz]
P_Tx = lambda f: 1.0*((f>=fTx-fBW/2) & (f<=fTx+fBW/2)); # Pulse Spectrum
P_Tx_f = P_Tx(f); # Pulse Definition
# Only Keep Positive Frequencies within Passband
passband_f_idx = np.argwhere(np.logical_and(P_Tx_f>pulse_cutoff, f>0)).flatten();
f = f[passband_f_idx]; P_Tx_f = P_Tx_f[passband_f_idx];
P_Tx_f = np.ones(P_Tx_f.shape); # Assume Flat Passband
# Get Receive Channel Data in the Frequency Domain
P_Rx_f = np.fft.fftshift(np.fft.fft(rxdata_h, n=nt, axis=0), axes=0);
P_Rx_f = P_Rx_f[passband_f_idx,:,:]; del rxdata_h;
apod = optApod(no_elements, 1); # Optimal Apodization for Lag-One Correlation
APOD, F, _ = np.meshgrid(apod, f, np.arange(P_Rx_f.shape[2])); 
P_Rx_f = APOD * (P_Rx_f * np.exp(-1j*2*np.pi*F*time[0]));
rxdata_f = interp1d(xpos, np.transpose(P_Rx_f, axes=(1,0,2)), 
    kind='nearest', axis=0, bounds_error=False, fill_value=0)(x);
rxdata_f = rxdata_f.astype(np.csingle); del P_Rx_f, APOD;
# Pulsed-Wave Frequency Response on Transmit
apod = np.eye(no_elements); delay = np.zeros((no_elements,no_elements));
txdata_f = np.zeros((x.size, f.size, tx_elmts.size), dtype=np.csingle);
for tx_idx in np.arange(tx_elmts.size): 
    # Construct Transmit Responses for Each Element
    apod_x = interp1d(xpos, apod[tx_idx,:], kind='nearest', bounds_error=False, fill_value=0)(x);
    delayIdeal = interp1d(xpos, delay[tx_idx,:], kind='nearest', bounds_error=False, fill_value=0)(x);
    txdata_f[:,:,tx_idx] = np.outer(apod_x, P_Tx_f)*np.exp(-1j*2*np.pi*np.outer(delayIdeal,f));
# Inverse Fourier Transform to Move from Frequency to Time Axis
t_delays = np.linspace(-2/(2*fTx), 2/(2*fTx), 101);
TD, F = np.meshgrid(t_delays, f);
FT = np.exp(1j*2*np.pi*F*TD);
# Direction of Arrival Smoothing
I = np.eye(no_elements);
F = I[:-3,:]-3*I[1:-2,:]+3*I[2:-1,:]-I[3:,:];
roughening = (I+reg_smoothing*np.dot(F.T, F));
invroughening = np.linalg.inv(roughening)

## Create Path Length Matrix
# Aberration Measurement Points in Medium
Nelem = xpos.size;
x_msmts = xpos[int(np.round(dwnsmp/2)-1)::dwnsmp];
z_msmts = np.arange(dmin, dmax, dwnsmp*pitch);
X_MSMTS, Z_MSMTS = np.meshgrid(x_msmts, z_msmts);
# Form List of Path Length Matrices for Each Element
path_indices = list(); pixel_indices = list(); path_lens = list();
for elmt_idx in np.arange(Nelem):
    start_time = timer.time()
    path_idx = np.array([]); pixel_idx = np.array([]); path_lengths = np.array([]);
    for meas_idx in np.arange(X_MSMTS.size):
        # Find Line-Pixel Intersection
        start_pt = np.array([xpos[elmt_idx], 0]); 
        end_pt = np.array([X_MSMTS.flatten(order='F')[meas_idx], 
            Z_MSMTS.flatten(order='F')[meas_idx]]);
        grid, intersegments = \
            line_pixel_intersection(x_grid, z_grid, start_pt, end_pt);
        # Populate Path-Length Matrix
        path_idx = np.hstack((path_idx, elmt_idx * X_MSMTS.size + meas_idx * \
            np.ones(intersegments['lengths']['val'].flatten().size))).astype(int);
        pixel_idx = np.hstack((pixel_idx, intersegments['lengths']['row'].flatten() + 
            Nz*intersegments['lengths']['col'].flatten())).astype(int);
        path_lengths = np.hstack((path_lengths, 
            intersegments['lengths']['val'].flatten())).astype(np.single);
    path_indices.append(path_idx); pixel_indices.append(pixel_idx); path_lens.append(path_lengths);
    print('Sparse Matrix Element '+str(elmt_idx)+' Completed'); 
    end_time = timer.time(); print(str(end_time-start_time)+' seconds');
# Assemble Full Path-Length Matrix
path_idx = np.hstack(path_indices); 
pixel_idx = np.hstack(pixel_indices);
path_lengths = np.hstack(path_lens);
path_len_mat = coo_matrix((path_lengths, 
    (path_idx,pixel_idx)), shape=(Nelem*X_MSMTS.size,Nx*Nz))
del path_idx, pixel_idx, path_lengths

## Iterative Sound Speed Reconstruction and Aberration Correction
# Create Image and Gain Compensation Maps
aberr_delay = np.zeros((z.size, x.size, no_elements-1), dtype=np.single);
img = np.zeros((z.size, x.size, no_elements), dtype=np.csingle);
img[0,:,:] = np.sum(txdata_f * np.conj(rxdata_f), axis=1);
# Keep the Original Transducer Signals
rxdata_f_orig = rxdata_f;
txdata_f_orig = txdata_f;
# Image and Sound Speed Reconstructions to Save at Each Iteration
IMG_iter = np.zeros(Z_IMG.shape+(num_sos_iterations,), dtype=np.csingle);
Crecon_iter = np.zeros(Zrecon.shape+(num_sos_iterations,), dtype=np.csingle); 
# Sound Speed and Aberration Correction Iterations
for sos_iter in np.arange(num_sos_iterations):
    ## Propagate Ultrasound Signals, Reconstruct Image, and Measure Time Shifts
    # Restore the Original Transducer Signals
    rxdata_f = rxdata_f_orig; 
    txdata_f = txdata_f_orig;
    # Propagate Ultrasound Signals in Depth
    for z_idx in np.arange(z.size-1):
        # Collect Sound Speed at This Depth
        clayer = interp2d(xrecon, zrecon, Crecon, kind='cubic', 
            bounds_error=False, fill_value=cbfm)(x, np.mean(z[z_idx:z_idx+2]))
        # Propagate Signals in Depth
        rxdata_f, txdata_f = propagate(x, z[z_idx], z[z_idx+1], 
            clayer, f, rxdata_f, txdata_f, aawin);
        # Compute Image at this Depth
        img[z_idx+1,:,:] = np.sum(txdata_f * np.conj(rxdata_f), axis=1);
        # Compute Aberration Delay Using Fourier Domain Auto-Correlation Method
        img_full = invroughening @ np.reshape(np.dot(np.reshape(np.transpose(txdata_f * np.conj(rxdata_f), axes=(0,2,1)), 
            newshape = (x.size*no_elements, f.size)), FT), newshape = (x.size, no_elements, t_delays.size)); 
        freq_estim = np.angle(np.mean(img_full[:,:,1::] * 
            np.conj(img_full[:,:,:-1:])))/(2*np.pi*np.mean(np.diff(t_delays)));
        aberr_delay[z_idx+1,:,:] = np.angle(np.mean(img_full[:,1::,:] * 
            np.conj(img_full[:,:-1:,:]),axis=2))/(2*np.pi*freq_estim);
        # Setup Next Depth Step
        print('z = '+str(z[z_idx])+' m / '+str(dov)+' m');
    # Coherently Sum Across Transmits
    img_recon = np.sum(img, axis=2); 
    # Sinc Interpolation to Upsample Ultrasound Image
    x_upsamp = x[0] + np.arange(3*x.size)*np.mean(np.diff(x))/3;
    X_GRID, Z_GRID = np.meshgrid(x_upsamp, z);
    img_recon = np.fft.ifft(np.fft.ifftshift(np.pad(np.fft.fftshift(np.fft.fft(img_recon, axis=1), axes=1), 
        ((0,0),(img_recon.shape[1],img_recon.shape[1]))), axes=1), axis=1); 
    IMG = (interp2d(x_upsamp, z, np.real(img_recon), kind='cubic')(x_img, z_img) + 
        1j*interp2d(x_upsamp, z, np.imag(img_recon), kind='cubic')(x_img, z_img));
    IMG_iter[:,:,sos_iter] = IMG;
    plt.figure(figsize=(9,6))
    imagesc(1000*x_img, 1000*z_img, db(np.abs(IMG)/np.max(np.abs(IMG))), dBrange);
    plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]'); 
    plt.title('Full Waveform Reconstruction'); 
    plt.savefig('figures/BMode'+str(sos_iter)+'.png'); 
    ## Reconstruct Sound Speed from Measurements
    # Construct Aberration Measurements
    Xmeas, Zmeas = np.meshgrid(x, z);
    aberration_times = np.zeros((z_msmts.size, x_msmts.size, nTx-1), dtype=np.single);
    aberration_mask = np.zeros((z_msmts.size, x_msmts.size, nTx-1), dtype=bool);
    distance = np.zeros((z_msmts.size, x_msmts.size, nTx-1), dtype=np.single);
    for xpos_idx in np.arange(nTx-1):
        # Median Filter 
        aberration_times[:,:,xpos_idx] = interp2d(x, z, 
            medfilt2d(aberr_delay[:,:,xpos_idx], kernel_size=medfilt_kernel))(x_msmts, z_msmts);
        # Remove Measurements Outside of F-Number Window and Start/End Depth
        aberration_mask[:,:,xpos_idx] = (Z_MSMTS > 
            Fnum*np.sqrt((X_MSMTS-np.mean(xpos[xpos_idx:xpos_idx+2]))**2+(dmin/Fnum/2)**2));
        distance[:,:,xpos_idx] = np.sqrt((X_MSMTS-np.mean(xpos[xpos_idx:xpos_idx+2]))**2+Z_MSMTS**2);
    # Following Section Relates Measurements to a Linearized Forward Model
    if sos_iter == 0: # Show the Following Images Only on the First Iteration
        xpos_idx = 64;
        # Visualize Differential Travel Times
        plt.figure(figsize=(9,4))
        imagesc(1000*x_msmts, 1000*z_msmts, aberration_times[:,:,xpos_idx-1], 
            [np.min(aberration_times[:,:,xpos_idx-1]), np.max(aberration_times[:,:,xpos_idx-1])]);
        plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]'); plt.title('Aberration Delay [s]'); plt.colorbar();
        plt.plot(1000*x, 1000*Fnum*np.sqrt((x-np.mean(xpos[xpos_idx-1:xpos_idx+1]))**2 + (dmin/Fnum/2)**2), 'r', linewidth=2); 
        plt.xlim(1000*np.array([np.min(x_msmts), np.max(x_msmts)])); 
        plt.ylim(1000*np.array([np.min(z_msmts), np.max(z_msmts)])); 
        plt.gca().invert_yaxis(); 
        plt.savefig('figures/AberrationDelayMeasurementL12_3v.png'); 
    # Measured Aberration Delays
    y = aberration_times.flatten(order='F')[aberration_mask.flatten(order='F')];
    # Linear Operators for Travel Time Tomography
    path_integrate = lambda ds: path_len_mat @ ds;
    backproject = lambda times: path_len_mat.T @ times;
    diff_times = (lambda times:  
        times[X_MSMTS.size:X_MSMTS.size*Nelem] - 
        times[:X_MSMTS.size*(Nelem-1)]);
    diff_times_adj = (lambda diff_times:
        np.vstack( (-diff_times[:X_MSMTS.size, np.newaxis], 
        diff_times[:X_MSMTS.size*(Nelem-2), np.newaxis] - 
        diff_times[X_MSMTS.size:X_MSMTS.size*(Nelem-1), np.newaxis],
        diff_times[X_MSMTS.size*(Nelem-2):X_MSMTS.size*(Nelem-1), np.newaxis]) ));
    mask = lambda diff_times: diff_times[aberration_mask.flatten(order='F')];
    mask_adj = lambda diff_times_mask: spray(diff_times_mask,aberration_mask.flatten(order='F'));
    # Forward/Adjoint Matrix Model
    H = lambda ds: mask(diff_times(path_integrate(ds)));
    HT = lambda dt: backproject(diff_times_adj(mask_adj(dt)));
    # Prior Covariance Matrix Q as a Linear Operator
    Qblur = lambda x: np.reshape(convolve2d(np.reshape(x, (Nz, Nx), order='F'), 
        np.outer(blur_z, blur_x), mode='same'), (Nz*Nx, 1), order='F');    
    Qlayers = lambda x: np.reshape(np.tile(convolve(np.sum(np.reshape(x, (Nz, Nx), order='F'), axis=1), 
        blur_z, mode='same'), (1, Nx)), (Nz*Nx, 1), order='F');
    Q = lambda x: alpha*Qblur(x) + (1-alpha)*Qlayers(x);
    # Observation Covariance Matrix R as Sparse Matrix
    W = mask((distance.flatten(order='F')/np.max(distance.flatten(order='F')))**(2));
    R = reg*W;
    # Define A (Symmetric) Matrix as Linear Operator
    HQHTpR = lambda x: H(Q(HT(x))).flatten()+R*x;
    HQHTpR_LinearOperator = LinearOperator((R.size,R.size), matvec=HQHTpR, rmatvec=HQHTpR)
    # Solve Using Conjugate Gradient
    xi, info = cg(HQHTpR_LinearOperator, y, tol=tol, atol=tol*np.linalg.norm(y), maxiter=max_iter);
    ds = Q(HT(xi));
    # Reconstruct Speed of Sound
    Crecon = 1/(1/Crecon + np.reshape(ds,(Nz,Nx),order='F'));
    Crecon_iter[:,:,sos_iter] = Crecon;
    # Show Reconstructed Image of Slowness
    plt.figure(figsize=(9,4))
    imagesc(1000*xrecon, 1000*zrecon, Crecon, [np.min(Crecon),np.max(Crecon)]); 
    plt.plot(1000*xpos, np.zeros(xpos.shape), 'w.', linewidth=2);
    plt.plot(1000*X_MSMTS, 1000*Z_MSMTS, 'r.', linewidth=2);
    plt.colorbar(); plt.xlabel('Lateral [mm]'); plt.ylabel('Axial [mm]'); 
    plt.title('Reconstructed Speed of Sound');  
    plt.savefig('figures/SoS'+str(sos_iter)+'.png'); 
