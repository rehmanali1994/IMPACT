function [rxdata_z2_f, txdata_z2_f] = ...
    propagate(x, z1, z2, c, f, rxdata_z1_f, txdata_z1_f, aawin)
% 
% PROPAGATE - Angular Spectrum Propagation of TX/RX Signals into the Medium
%
% This function propagates transmit and receive wavefields at from one 
% depth to another depth using the angular spectrum method
% 
% INPUTS:
% x                  - 1 x X vector of x-grid positions for wavefield
% z1                 - depth of input TX and RX wavefields
% z2                 - depth of output TX and RX wavefields
% c                  - 1 x X vector of speed of sound [m/s] between z1 and z2
% f                  - 1 x F vector of pulse frequencies in spectrum
% rxdata_z1_f        - X x F x N array of input RX wavefields at z1
% txdata_z1_f        - X x F x N array of input TX wavefields at z1
% aawin              - 1 x X vector of lateral taper to prevent wraparound
% 
% OUTPUT:
% rxdata_z2_f        - X x F x N array of output RX wavefields at z2
% txdata_z2_f        - X x F x N array of output TX wavefields at z2
% 

% Verify the Number of Common Shot Gathers
ns = size(txdata_z1_f, 3); assert(size(rxdata_z1_f, 3) == ns, ...
    'Number of sources must equal to number of common-source gathers');
AAwin = repmat(aawin(:), [1, numel(f), ns]);

% Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
ft = @(sig) fftshift(fft(AAwin.*sig, [], 1), 1);
ift = @(sig) AAwin.*ifft(ifftshift(sig, 1), [], 1);

% Spatial Grid
dx = mean(diff(x)); nx = numel(x); 

% FFT Axis for Lateral Spatial Frequency
kx = mod(fftshift((0:nx-1)/(dx*nx))+1/(2*dx), 1/dx)-1/(2*dx);

% Mean Slowness and Lateral Slowness Variation [s/m] between z1 and z2
s = 1./c; % Slowness [s/m]
smean = mean(s); % Mean Slowness [s/m]
ds = s - smean; % Lateral Variation in Slowness [s/m] Along x

% Continuous Wave Response By Downward Angular Spectrum
[F, Kx] = meshgrid(f,kx); % Create Grid in f-kx
[~, Ds] = meshgrid(f,ds); % Create Grid
Kz = sqrt((F*smean).^2-Kx.^2); % Axial Spatial Frequency
H = exp(1i*2*pi*Kz*(z2-z1)); % Propagation Filter
H(Kz.^2 <= 0) = 0; % Remove Evanescent Components
H = repmat(H, [1,1,ns]); % Replicate Across Shots
dH = exp(1i*2*pi*F.*Ds*(z2-z1)); % Phase Screen
dH = repmat(dH, [1,1,ns]); % Replicate Across Shots

% Apply Propagation Filter
rxdata_z2_f = dH.*ift(H.*ft(rxdata_z1_f));
txdata_z2_f = conj(dH).*ift(conj(H).*ft(txdata_z1_f)); 

end