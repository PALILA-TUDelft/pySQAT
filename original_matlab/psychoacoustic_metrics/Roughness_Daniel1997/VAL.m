% test_dw_reference.m
% -----------------------------------------------
%  Build the Daniel & Weber reference signal
%  and run the roughness model for a quick sanity-check
% -----------------------------------------------

%% 1.  Signal parameters
fs        = 48e3;          % sample rate (Hz) – matches model default
dur       = 2.0;           % seconds
f_carrier = 1e3;           % 1 kHz tone
f_mod     = 70;            % 70 Hz modulation
spl       = 60;            % level in dB SPL

%% 2.  Generate the AM tone (mono, Pascals)
t      = (0:1/fs:dur-1/fs).';                 % time vector (column)
p_rms  = 20e-6 * 10^(spl/20);                 % 60 dB SPL → 0.02 Pa RMS
A      = p_rms * sqrt(2);                     % sine RMS→peak
env    = 0.5 * (1 + sin(2*pi*f_mod*t));       % 100 % AM envelope
insig  = A .* env .* sin(2*pi*f_carrier*t);   % final stimulus (Pa)

%% 3.  Run the Daniel & Weber roughness model
time_skip = 0;     % don’t exclude any start-up portion
show      = true;  % display plots

OUT = Roughness_Daniel1997(insig, fs, time_skip, show);

%% 4.  Print a few key results
fprintf('\nReference-tone sanity check (should be ~1 asper)\n');
fprintf('  Mean roughness : %.3f asper\n', OUT.Rmean);
fprintf('  Max  roughness : %.3f asper\n', OUT.Rmax);
fprintf('  Min  roughness : %.3f asper\n\n', OUT.Rmin);
