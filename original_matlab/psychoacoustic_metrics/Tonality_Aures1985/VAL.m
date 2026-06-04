%% 1. PARAMETERS
fs       = 44100;             % Sampling rate in Hz
duration = 1.0;               % Signal duration in seconds
f0       = 1000;              % Tone frequency in Hz
Lp       = 60;                % Desired SPL in dB
pref     = 20e-6;             % Reference pressure in Pa

%% 2. TIME VECTOR
t = 0 : 1/fs : duration - 1/fs;  % Time vector

%% 3. SIGNAL GENERATION
rms_target = pref * 10^(Lp / 20);     % RMS amplitude for desired SPL
amp        = rms_target * sqrt(2);    % Peak amplitude for sine wave
signal     = amp * sin(2 * pi * f0 * t);  % Generate pure tone

%% 4. TONALITY CALCULATION
LoudnessField = 0;       % 0 = free field
time_skip     = 0.5;     % Skip first 0.5 seconds
show_plot     = true;    % Plot results

OUT = Tonality_Aures1985(signal, fs, LoudnessField, time_skip, show_plot);

%% 5. PRINT RESULTS
fprintf('\nTonality Statistics:\n');
fprintf('  InstantaneousTonality : %.3f\n', mean(OUT.InstantaneousTonality));
fprintf('  LoudnessWeighting     : %.3f\n', mean(OUT.LoudnessWeighting));
fprintf('  TonalWeighting        : %.3f\n', mean(OUT.TonalWeighting));