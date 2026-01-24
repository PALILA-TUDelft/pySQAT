fname = [basepath_SQAT 'sound_files' filesep 'reference_signals' filesep 'RefSignal_Loudness_ISO532_1.wav'];
[insig,fs] = audioread(fname);
dBFS = 94; % A priori knowledge
[OB_filt,fc] = Do_OB13_ISO532_1(insig,fs);
% Sub example: obtaining the calibrated RMS level of each band:
OB_lvls = 20*log10(rms(OB_filt))+dBFS;

figure;
semilogx(fc,OB_lvls,'bo-'); hold on;
set(gca,'XLim',[min(fc)-1 max(fc)+1000]);
set(gca,'XTick',fc);
set(gca,'XTickLabel',round(fc));
xlabel('Frequency (Hz)');
ylabel('Band level (dB SPL)');

% Total level of the input signal:
lvl_orig = 20*log10(rms(insig))+dBFS;
fprintf('Level as obtained directly from the input signal=%.1f dB SPL\n',lvl_orig);

% The sum of the power of each band leads to the total level:
lvl_from_outsig = 10*log10(sum(10.^(OB_lvls/10))); % '/10' means squared, therefore it is 10*log10()
fprintf('Level obtained from the filtered signals=%.1f dB SPL\n',lvl_from_outsig);