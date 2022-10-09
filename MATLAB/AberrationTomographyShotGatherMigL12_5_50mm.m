clear
clc

% Load all Functions from Subdirectories
addpath(genpath(pwd));

% Load File and Original k-Wave Simulation Grid
load_filename = '../Datasets/PhantomL12-5-50mm.mat';
load(load_filename);
[nT, nRx, nTx] = size(fsr_dataset_fund); 
tx_elmts = 1:1:nTx;
rxdata_h = fsr_dataset_fund(:,:,tx_elmts); 
clearvars fsr_dataset_fund;

%% Define Parameters
dov = 46e-3; % Max Depth [m]
upsamp_x = 2; % Upsampling in x (assuming dx = pitch)
upsamp_z = 1; % Upsampling in z (assuming dz = pitch)
Nx0 = 320; % Number of Points Laterally in x
cbfm = 1540; % initial sound speed [m/s]
fTx = 6e6; % frequency [Hz]
fBW = 6e6; % bandwidth at half maximum [Hz]
pulse_cutoff = 1e-3; % passband cutoff
ord = 100; % Strength of Anti-Aliasing Window 
% Parameters to Filter Time Delay Measurements
Fnum = 1.0; % f-number to limit acceptance angle
dmin = 0.004; % start depth [m]
dmax = 0.045; % end depth [m]
dwnsmp = 5; % downsample measurements for tomography
medfilt_kernel = [9,9]; % Size of Median Filter
% B-Mode Image Parameters/Imaging Window
dBrange = [-80, 0]; % Dynamic Range [decibels]
Nximg = 1001; Nzimg = 1001;
xlims = [-25e-3, 25e-3]; 
zlims = [2e-3, 45e-3];
% Sound Speed Reconstruction Parameters
blur_z = gausswin(100, 4); % Z - Blurring Over Reconstruction Grid
blur_x = gausswin(100, 4); % X - Blurring Over Reconstruction Grid
max_iter = 200; tol = 1e-3; % Conjugate Gradient Parameters
reg = 1e-2; % Regularization Corresponding to Relative Time of Flight Error
alpha = 1.0; % Relative Weighting of Noninformative Prior to Layered Prior
    % 1.0 for full noninformative prior; 0.0 for fully layered prior
num_sos_iterations = 10; % Number of Sound Speed Estimations/Iterations

%% Create Simulation, Image Reconstruction, and Sound Speed Grids
lambda = cbfm/fTx; % wavelength [m]
pitch = mean(diff(rxAptPos(:,1))); % element spacing [m]
no_elements = size(rxAptPos,1); % number of elements
xpos = (-(no_elements-1)/2:(no_elements-1)/2)*pitch; % element position [m]
% Grid for Simulation
x = (-(upsamp_x*Nx0-1)/2:(upsamp_x*Nx0-1)/2)*(pitch/upsamp_x); % m
Nu1 = round(dov/((lambda/2)/upsamp_z)); 
z = ((0:Nu1-1))*(lambda/2)/upsamp_z; % m
xmax = (max(abs(xpos))+max(abs(x)))/2; % Lateral Cutoff for Anti-Aliasing [m]
aawin = 1./sqrt(1+(x/xmax).^ord); % Anti-Aliasing Window
% Grid for Image Reconstruction
x_img = linspace(xlims(1), xlims(2), Nximg);
z_img = linspace(zlims(1), zlims(2), Nzimg);
[X_IMG, Z_IMG] = meshgrid(x_img, z_img);
% Grid For Sound Speed [m/s] Estimate
dx = mean(diff(x)); Nx = numel(x); Nz = ceil((z(end)-z(1))/dx)+1; 
x_grid.start = x(1); x_grid.spacing = dx; x_grid.N = Nx;
z_grid.start = z(1); z_grid.spacing = dx; z_grid.N = Nz;
Crecon = cbfm*ones(Nz, Nx); % Initial Speed of Sound
xrecon = x_grid.start + x_grid.spacing*(0:x_grid.N-1);
zrecon = z_grid.start + z_grid.spacing*(0:z_grid.N-1);
[Xrecon, Zrecon] = meshgrid(xrecon, zrecon);

%% Frequency-Domain Representation of Transmitted and Received Wavefields
% Transmit Impulse Response in Frequency Domain
nt = numel(time); % [s]
fs = 1/mean(diff(time)); % [Hz] 
f = (fs/2)*(-1:2/nt:1-2/nt); % [Hz]
P_Tx = @(f) 1.0*((f>=fTx-fBW/2) & (f<=fTx+fBW/2)); % Pulse Spectrum
P_Tx_f = P_Tx(f); % Pulse Definition
% Only Keep Positive Frequencies within Passband
passband_f_idx = find((P_Tx_f > pulse_cutoff) & (f > 0));
f = f(passband_f_idx); P_Tx_f = P_Tx_f(passband_f_idx);
P_Tx_f = ones(size(P_Tx_f)); % Assume Flat Passband
% Get Receive Channel Data in the Frequency Domain
P_Rx_f = fftshift(fft(rxdata_h, nt, 1), 1);
P_Rx_f = P_Rx_f(passband_f_idx,:,:); clearvars rxdata_h;
apod = optApod(no_elements, 1); % Optimal Apodization for Lag-One Correlation
[APOD, F, ~] = meshgrid(apod, f, 1:size(P_Rx_f,3)); 
P_Rx_f = APOD .* (P_Rx_f .* exp(-1i*2*pi*F*time(1)));
rxdata_f = interp1(xpos, permute(P_Rx_f, [2,1,3]), x, 'nearest', 0);
% Pulsed-Wave Frequency Response on Transmit
apod = eye(no_elements); delay = zeros(no_elements);
txdata_f = zeros(numel(x), numel(f), numel(tx_elmts));
for tx_idx = 1:numel(tx_elmts) 
    % Construct Transmit Responses for Each Element
    apod_x = interp1(xpos, apod(tx_idx,:), x, 'nearest', 0);
    delayIdeal = interp1(xpos, delay(tx_idx,:), x, 'nearest', 0);
    txdata_f(:,:,tx_idx) = (apod_x'*P_Tx_f).*exp(-1i*2*pi*delayIdeal'*f);
end
% Inverse Fourier Transform to Move from Frequency to Time Axis
t_delays = linspace(-2/(2*fTx), 2/(2*fTx), 101);
[TD, F] = meshgrid(t_delays, f);
FT = exp(1i*2*pi*F.*TD);

%% Create Path Length Matrix
% Aberration Measurement Points in Medium
Nelem = numel(xpos);
x_msmts = xpos(round(dwnsmp/2):dwnsmp:end);
z_msmts = dmin:(dwnsmp*pitch):dmax;
[X_MSMTS, Z_MSMTS] = meshgrid(x_msmts, z_msmts);
% Form Cell Array of Path Length Matrices for Each Element
path_length_matrix = cell(Nelem,1);
for elmt_idx = 1:Nelem
    tic; path_idx = []; pixel_idx = []; path_lengths = [];
    for meas_idx = 1:numel(X_MSMTS)
        % Find Line-Pixel Intersection
        start_pt = [xpos(elmt_idx), 0]; 
        end_pt = [X_MSMTS(meas_idx), Z_MSMTS(meas_idx)];
        [grid, intersegments] = ...
            line_pixel_intersection(x_grid, z_grid, start_pt, end_pt);
        % Populate Path-Length Matrix
        path_idx = [path_idx; meas_idx * ...
            ones(numel(intersegments.lengths.val(:)),1)];
        pixel_idx = [pixel_idx; intersegments.lengths.row(:) + ...
            (intersegments.lengths.col(:)-1)*Nz];
        path_lengths = [path_lengths; intersegments.lengths.val(:)];
    end
    path_length_matrix{elmt_idx} = sparse(path_idx, pixel_idx, ...
        path_lengths, numel(X_MSMTS), Nx*Nz);
    disp(['Sparse Matrix Element ', num2str(elmt_idx), ' Completed']); toc;
end
% Assemble Full Path-Length Matrix
path_len_mat = sparse(Nelem*numel(X_MSMTS), Nx*Nz);
for elmt_idx = 1:Nelem
    tic; path_len_mat(numel(X_MSMTS)*(elmt_idx-1)+...
        (1:numel(X_MSMTS)),:) = path_length_matrix{elmt_idx};
    disp(['Sparse Matrix Element ', num2str(elmt_idx), ' Assembled']); toc;
end

%% Iterative Sound Speed Reconstruction and Aberration Correction
% Create Image and Gain Compensation Maps
aberr_delay = zeros(numel(z), numel(x), no_elements-1);
img = zeros(numel(z), numel(x), no_elements);
img(1,:,:) = sum(txdata_f .* conj(rxdata_f),2);
% Keep the Original Transducer Signals
rxdata_f_orig = rxdata_f;
txdata_f_orig = txdata_f;
% Image and Sound Speed Reconstructions to Save at Each Iteration
IMG_iter = zeros([size(Z_IMG), num_sos_iterations]);
Crecon_iter = zeros([size(Zrecon), num_sos_iterations]);
% Sound Speed and Aberration Correction Iterations
for sos_iter = 1:num_sos_iterations
    %% Propagate Ultrasound Signals, Reconstruct Image, and Measure Time Shifts
    % Restore the Original Transducer Signals
    rxdata_f = rxdata_f_orig;
    txdata_f = txdata_f_orig;
    % Propagate Ultrasound Signals in Depth
    for z_idx = 1:numel(z)-1
        % Collect Sound Speed at This Depth
        clayer = interp2(Xrecon, Zrecon, Crecon, ...
            x, mean(z(z_idx:z_idx+1)), 'spline', cbfm);
        % Propagate Signals in Depth
        [rxdata_f, txdata_f] = propagate(x, z(z_idx), z(z_idx+1), ...
            clayer, f, rxdata_f, txdata_f, aawin);
        % Compute Image at this Depth
        img(z_idx+1,:,:) = sum(txdata_f .* conj(rxdata_f),2);
        % Compute Aberration Delay Using Fourier Domain Auto-Correlation Method
        img_full = reshape(reshape(permute(txdata_f .* conj(rxdata_f), [1,3,2]), ...
            [numel(x)*no_elements, numel(f)])*FT, [numel(x), no_elements, numel(t_delays)]);    
        freq_estim = angle(mean(img_full(:,:,2:end) .* ...
            conj(img_full(:,:,1:end-1)),[1,2,3]))/(2*pi*mean(diff(t_delays)));
        aberr_delay(z_idx+1,:,:) = angle(mean(img_full(:,2:end,:) .* ...
            conj(img_full(:,1:end-1,:)),3))./(2*pi*freq_estim);
        % Setup Next Depth Step
        disp(['z = ', num2str(z(z_idx)), ' m / ', num2str(dov), ' m']);
    end
    % Coherently Sum Across Transmits
    img_recon = sum(img, 3); 
    % Sinc Interpolation to Upsample Ultrasound Image
    x_upsamp = x(1) + (0:3*numel(x)-1)*mean(diff(x))/3;
    [X_GRID, Z_GRID] = meshgrid(x_upsamp, z);
    img_recon = ifft(ifftshift(padarray(fftshift(fft(img_recon, [], 2), 2), ...
        [0, size(img_recon, 2)], 0, 'both'), 2), [], 2);
    IMG = interp2(X_GRID, Z_GRID, img_recon, X_IMG, Z_IMG, 'spline');
    IMG_iter(:,:,sos_iter) = IMG;
    % Display Ultrasound Image
    figure; imagesc(1000*x_img, 1000*z_img, ...
        db(abs(IMG)/max(abs(IMG(:)))), dBrange);
    xlabel('Lateral [mm]'); ylabel('Axial [mm]'); 
    title('Full Waveform Reconstruction'); 
    zoom on; axis equal; axis xy; axis image; 
    colormap gray; colorbar(); set(gca, 'YDir', 'reverse'); 
    %% Reconstruct Sound Speed from Measurements
    % Construct Aberration Measurements
    [Xmeas, Zmeas] = meshgrid(x, z);
    aberration_times = zeros(numel(z_msmts), numel(x_msmts), nTx-1);
    aberration_mask = false(numel(z_msmts), numel(x_msmts), nTx-1);
    distance = zeros(numel(z_msmts), numel(x_msmts), nTx-1);
    for xpos_idx = 1:nTx-1
        % Median Filter
        aberration_times(:,:,xpos_idx) = interp2(Xmeas, Zmeas, ...
            medfilt2(aberr_delay(:,:,xpos_idx),medfilt_kernel), X_MSMTS, Z_MSMTS);
        % Remove Measurements Outside of F-Number Window and Start/End Depth
        aberration_mask(:,:,xpos_idx) = (Z_MSMTS > ...
            Fnum*sqrt((X_MSMTS-mean(xpos(xpos_idx:xpos_idx+1))).^2 + (dmin/Fnum/2).^2));
        distance(:,:,xpos_idx) = sqrt((X_MSMTS-mean(xpos(xpos_idx:xpos_idx+1))).^2 + Z_MSMTS.^2);
    end
    % Following Section Relates Measurements to a Linearized Forward Model
    if sos_iter == 1 % Show the Following Images Only on the First Iteration
        xpos_idx = 128;
        % Visualize Differential Travel Times
        figure; imagesc(1000*x_msmts, 1000*z_msmts, aberration_times(:,:,xpos_idx), ...
            [min(min(aberration_times(:,:,xpos_idx))), max(max(aberration_times(:,:,xpos_idx)))]);
        xlabel('Lateral [mm]'); ylabel('Axial [mm]'); title('Aberration Delay [s]'); colorbar; hold on;
        plot(1000*x, 1000*Fnum*sqrt((x-mean(xpos(xpos_idx:xpos_idx+1))).^2 + (dmin/Fnum/2).^2), 'r', 'Linewidth', 2); 
        axis image; xlim(1000*[min(x_msmts(:)), max(x_msmts(:))]); ylim(1000*[min(z_msmts(:)), max(z_msmts(:))]); 
    end
    % Measured Aberration Delays
    y = aberration_times(aberration_mask(:));
    % Linear Operators for Travel Time Tomography
    path_integrate = @(ds) path_len_mat*ds;
    backproject = @(times) path_len_mat'*times;
    diff_times = @(times) ...
        times(1+numel(X_MSMTS):numel(X_MSMTS)*Nelem) - ...
        times(1:numel(X_MSMTS)*(Nelem-1));
    diff_times_adj = @(diff_times) ...
        [-diff_times(1:numel(X_MSMTS)); ...
        diff_times(1:numel(X_MSMTS)*(Nelem-2)) - ...
        diff_times(1+numel(X_MSMTS):numel(X_MSMTS)*(Nelem-1));
        diff_times(1+numel(X_MSMTS)*(Nelem-2):numel(X_MSMTS)*(Nelem-1))];
    mask = @(diff_times) diff_times(aberration_mask(:));
    mask_adj = @(diff_times_mask) spray(diff_times_mask,aberration_mask(:));
    % Forward/Adjoint Matrix Model
    H = @(ds) mask(diff_times(path_integrate(ds)));
    HT = @(dt) backproject(diff_times_adj(mask_adj(dt)));
    % Prior Covariance Matrix Q as a Linear Operator
    Qblur = @(x) reshape(conv2(blur_z, blur_x, ...
        reshape(x, [Nz, Nx]), 'same'), [Nz*Nx, 1]);
    Qlayers = @(x) reshape(repmat(conv( ...
        sum(reshape(x, [Nz, Nx]), 2), blur_z, 'same'), [1, Nx]), [Nz*Nx, 1]);
    Q = @(x) alpha*Qblur(x) + (1-alpha)*Qlayers(x);
    % Observation Covariance Matrix R as Sparse Matrix
    W = mask((distance(:)/max(distance(:))).^(2));
    R = reg*W;
    % Define A (Symmetric) Matrix as Linear Operator
    HQHTpR = @(x) H(Q(HT(x)))+R.*x;
    % Solve Using Conjugate Gradient
    xi = pcg(HQHTpR, y(:), tol, max_iter);
    ds = Q(HT(xi));
    % Reconstruct Speed of Sound
    Crecon = 1./(1./Crecon + reshape(ds,[Nz,Nx]));
    Crecon_iter(:,:,sos_iter) = Crecon;
    % Show Reconstructed Image of Slowness
    figure; imagesc(1000*xrecon, 1000*zrecon, Crecon); 
    colorbar; axis image; hold on;
    xlabel('Lateral [mm]'); ylabel('Axial [mm]'); 
    plot(1000*xpos, zeros(size(xpos)), 'w.', 'Linewidth', 2);
    plot(1000*X_MSMTS(:), 1000*Z_MSMTS(:), 'r.', 'Linewidth', 2);
    title('Reconstructed Speed of Sound'); getframe(gca);
end