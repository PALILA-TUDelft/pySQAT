
fc = 1000;
Lp = 60;
fs = 48000;
duration = 5;
p_ref = 20e-6;

p_rms = p_ref * (10^(Lp / 20));
amplitude = sqrt(2) * p_rms;

t = 0:1/fs:duration-1/fs;
signal = amplitude * sin(2 * pi * fc * t);

results = Tonality_ECMA418_2(insig, fs);