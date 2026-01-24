% generate test signal
fs = 48e3;  % sample rate
t = 0:1/fs:5-1/fs;
length(t)
insig = 0.5 * sin(2 * pi * 440 * t)' + 0.25 * sin(2 * pi * 880 * t)';
OUT = Tonality_ECMA418_2(insig, fs, 'free-frontal', 0.304, true);