function see(filePath)
%SEE  Inspect a WAV file: waveform, log–spectrogram, and audio playback.
%
%   SEE(FILEPATH) loads the WAV file at FILEPATH, converts stereo to mono
%   (if necessary), shows its time-domain waveform, a logarithmic-frequency
%   spectrogram (1024-point FFT, 50 % overlap), and plays the sound.
%
%   Example
%   -------
%       see('speech_sample.wav')

    % --- House-keeping ---
    close all;                                   % Close any open figures
    if ~isfile(filePath)
        error("File not found: %s", filePath);
    end

    % --- Load the WAV file ---
    [audio, Fs] = audioread(filePath);           % audio = NxC, Fs = Hz

    % Convert stereo (or multichannel) to mono by averaging columns
    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end

    % Time axis (s)
    t = (0:numel(audio)-1).' / Fs;

    % --- Plot waveform ---
    figure('Name', 'Audio inspection', 'NumberTitle', 'off', ...
           'Position', [100 100 1000 600]);

    subplot(2,1,1)
    plot(t, audio, 'LineWidth', 0.9);
    maxAmp = max(abs(audio));
    ylim([-maxAmp maxAmp]);
    xlim([0 t(end)]);
    grid on; grid minor;
    title('Waveform');
    xlabel('Time (s)');
    ylabel('Amplitude');

    % --- Plot spectrogram (log-frequency, 80-dB window) ----------------------
    subplot(2,1,2)
    
    window    = 1024;
    noverlap  = 512;
    nfft      = 1024;
    
    [S,F,T]   = spectrogram(audio, window, noverlap, nfft, Fs, "yaxis");
    P_dB      = 10*log10(abs(S).^2 + eps);
    
    dynRange  = 80;                                 % top 80 dB
    cLim      = [max(P_dB(:)) - dynRange,  max(P_dB(:))];
    
    imagesc(T, F, P_dB, cLim); axis xy
    set(gca, "YScale","log")
    ylim([F(2) F(end)])                             % avoid 0 Hz on log axis
    xlim([0 t(end)])
    
    % Robust colormap: viridis if you have it, else parula
    try   colormap("viridis")
    catch colormap("parula"), warning("Using 'parula' colormap (no viridis).")
    end
    colorbar
    
    grid on, grid minor
    title("Spectrogram"), xlabel("Time (s)"), ylabel("Frequency (Hz)")

end