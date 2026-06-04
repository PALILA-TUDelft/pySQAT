%% -----------------------------------------------------------------
%  Validation signal: broadband roar + 800 Hz tone (20 s, fs = 48 kHz)
% ------------------------------------------------------------------
fs          = 48e3;                     % Hz (filter bank expects 48 kHz)
dur_total   = 20;                       % s
tone_freq   = 800;                      % Hz  – blade passage / fan tone
spl_broad   = 90;                       % dB  – *peak* broadband level
spl_tone    = spl_broad - 20;           % dB  – tone 20 dB below overall
dBFS        = 94;                       % library assumes 1.0 ↔ 94 dB SPL

pref        = 2e-5;                     % 0 dB SPL reference (Pa)
FS_pa       = pref * 10^(dBFS/20);      % full-scale (RMS) in pascals

% Helper: convert desired RMS pressure (Pa) to ±1 full-scale amplitude
pa2lin = @(pa_rms) pa_rms / FS_pa;

%% 1) Time vector
t = (0 : 1/fs : dur_total-1/fs).';      % column vector

%% 2) Broadband component (white noise with half-cosine envelope)
rng(0);                                 % reproducible
white_raw   = randn(size(t));                       
white_raw   = white_raw / rms(white_raw);           % unit RMS

env         = sin(pi * t / dur_total);              % 0➜1➜0 half-cos
target_rms  = pref * 10^(spl_broad/20);             % Pa, broadband RMS
broadband   = env .* white_raw .* pa2lin(target_rms);

%% 3) Pure-tone component (steady 800 Hz)
tone_rms    = pref * 10^(spl_tone/20);              % Pa
tone        = pa2lin(tone_rms) * sin(2*pi*tone_freq*t);

%% 4) Assemble final clip (column vector, as required by EPNL code)
flyover = broadband + tone;            % units: “full-scale” (|x| ≤ 1)
flyover = flyover(:);                  % ensure [N×1]

%% 5) (Optional) run the validator itself
OUT = EPNL_FAR_Part36( ...
        flyover,  fs, 1, ...           % insig, fs, method = 1 (audio)
        0.5,      10, true);           % dt = 0.5 s, threshold = 10 dB, show plots

fprintf('\nEPNL of validation clip: %.2f EPNdB\n', OUT.EPNL);
