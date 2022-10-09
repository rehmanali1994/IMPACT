function apod = optApod(Nelem,Nlags,varargin)
%OPTAPOD Find apodization that maximizes short lag autocorrelation
% apod = optApod(Nelem,Nlags,tol,print)
%   Nelem = number of elements in aperture
%   Nlags = number of lags in optimization
%   tol = (optional) numerical tolerance for optimization (default: tol = eps)

% Handle Optional Numerical Tolerance Input
if nargin == 2, tol = eps; else, tol = varargin{1}; end

% Laplacian with Homogeneous BCs
LagNDiff = diag(2*Nlags*ones(Nelem,1));
for lag = 1:Nlags
    LagNDiff = LagNDiff - ...
        diag(ones(Nelem-lag,1),lag) - ...
        diag(ones(Nelem-lag,1),-lag);
end

% Solve Ball-Exclusion Constrained Optimization Problem
apod = randn(Nelem,1); % Initial Apodization
prev_lambda = 0; lambda = 1; % Lagrange Multiplier
while (abs(norm(apod)-1) > tol) && (abs((lambda-prev_lambda)/lambda) > tol)
    prev_lambda = lambda; % Update Previous Lagrange Multiplier
    apod = abs(apod)/norm(apod); % Project Onto Nonnegative Norm-2 Ball
    LagNDiffConstrained = [LagNDiff, apod; apod', 0];
    sol = LagNDiffConstrained\[zeros(Nelem,1);1];
    apod = sol(1:Nelem); % Extract Optimal Apodization
    lambda = sol(end); % New Lagrange Multiplier
end
    
end

