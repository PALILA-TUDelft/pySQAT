
validate = "Tonality_Aures1985";

switch validate

    %% LOUDNESS
    %-----------------------------------------------------------------------------------------------------------------------
    case "Loudness_ISO532_1"

        disp("Running Loudness_ISO532_1 test...")

        fs = 48000;
        duration = 5.0;
        f_tone = 1000;
        desired_spl = 40;
        a_rms_pa = 2e-5 * 10^(desired_spl/20);
        a_peak_pa = a_rms_pa * sqrt(2);
        fullscale_pa = 2e-5 * 10^(94/20);
        amplitude = a_peak_pa/fullscale_pa;
        t = linspace(0, duration, fs * duration);
        tone = amplitude * sin(2 * pi * f_tone * t);

        OUT = Loudness_ISO532_1( ...
            tone, ... % insig
            fs, ... % fs
            0, ... % field
            2, ... % method
            0, ... % time skip
            false); % show
        
        fprintf('Overall loudness (median of time-series): %.2f sone\n', median(OUT.InstantaneousLoudness));
        fprintf('5-percentile loudness N5:  %.2f sone\n', OUT.N5);
        fprintf('95-percentile loudness N95: %.2f sone\n', OUT.N95);
        fprintf('Loudness level (median):   %.2f phon\n', median(OUT.InstantaneousLoudnessLevel));

    case "EPNL_FAR_Part36"

        disp("Running EPNL_FAR_Part36 test...")

        fs          = 48000;
        dur_total   = 20.0;
        tone_freq   = 800.0;
        spl_broad   = 90.0;
        spl_tone    = spl_broad - 20;
        dBFS        = 94.0;
        pref        = 2e-5;
        FS_pa       = pref * 10^(dBFS/20);
        pa_to_linear = @(pa_rms) pa_rms / FS_pa;
        
        t           = 0:1/fs:dur_total - 1/fs;
        env         = sin(pi * t / dur_total);
        
        target_rms  = pref * 10^(spl_broad/20);
        white_raw   = randn(size(t));
        white_raw   = white_raw / sqrt(mean(white_raw.^2));
        white       = env .* white_raw * pa_to_linear(target_rms);
 
        tone_rms    = pref * 10^(spl_tone/20);
        tone        = pa_to_linear(tone_rms) * sin(2*pi*tone_freq*t);
        
        flyover     = (white + tone).';
        
        OUT = EPNL_FAR_Part36( ...
            flyover, ...
            fs, ...
            1, ...
            0.5, ...
            10, ...
            false ...
        );
        % 1.0015 and 1.001139
        % 1.0003 and 1.00029
        fprintf('EPNL of validation clip: %.2f EPNdB\n', OUT.EPNL);

    %% ROUGHNESS
    %-----------------------------------------------------------------------------------------------------------------------

    case "Roughness_Daniel1997"

        disp("Running Roughness_Daniel1997 test...")


        % Signal generation and roughness analysis
        fs = 48000;            % sampling rate (Hz)
        dur = 5.0;             % seconds
        f_carrier = 1000.0;    % Hz
        f_mod = 70.0;          % Hz
        L_spl = 60.0;          % dB SPL
        
        t = 0:1/fs:dur-1/fs;   % time vector
        envelope = 0.5 * (1.0 + sin(2*pi*f_mod*t));      % 0…1
        p_rms = 20e-6 * 10^(L_spl/20);                   % 60 dB SPL → Pa
        A = p_rms / sqrt(mean(envelope.^2));             % <<< key line
        signal = A * envelope .* sin(2*pi*f_carrier*t);  % double precision
        
        OUT = Roughness_Daniel1997(single(signal), fs, ...
                                  0.0, true);
        
        fprintf('  Mean roughness  : %g asper\n', OUT.Rmean);
        fprintf('  Max  roughness  : %g asper\n', OUT.Rmax);
        fprintf('  10 %% exceedance : %g asper\n', OUT.R10);



    %% TONALITY
    %-----------------------------------------------------------------------------------------------------------------------

    case "Tonality_Aures1985"

        disp("Running Tonality_Aures1985 test...")

        fs = 48000;
        duration = 5.0;
        f0 = 1000;
        Lp = 60;
        pref = 20e-6;
        
        t = 0:1/fs:(duration - 1/fs);
        rms_target = pref * 10^(Lp / 20);
        amp = rms_target * sqrt(2);
        signal = amp * sin(2 * pi * f0 * t);

        result = Tonality_Aures1985( ...
            signal, ...
            fs, ...
            0, ...
            0.5, ...
            false);
        
        fprintf('Tonality Statistics:\n');
        fprintf('InstantaneousTonality: %.3g\n', mean(result.InstantaneousTonality));
        fprintf('LoudnessWeighting: %.3g\n', mean(result.LoudnessWeighting));
        fprintf('TonalWeighting: %.3g\n', mean(result.TonalWeighting));

    %% FLUCTUTATION STRENGTH
    %-----------------------------------------------------------------------------------------------------------------------

    case "FluctuationStrength_Osses2016"

        fs = 48000;         % Sampling frequency (Hz)
        duration = 5.0;     % Duration of signal (s)
        t = 0:1/fs:duration - 1/fs; % Time vector
        
        f_carrier = 1000;   % Carrier frequency (Hz)
        f_mod = 4;          % Modulation frequency (Hz)
        mod_index = 1.0;    % Modulation index
        
        % --- Generate AM tone ---
        signal = (1 + mod_index * sin(2 * pi * f_mod * t)) .* sin(2 * pi * f_carrier * t);
        
        % --- Set to 60 dB SPL (0.02 Pa RMS) ---
        rms_target = 0.02;
        rms_current = rms(signal); % MATLAB's rms function
        signal = signal * (rms_target / rms_current);
        
        % --- Run the FluctuationStrength_Osses2016 function ---
        % Ensure FluctuationStrength_Osses2016.m and its dependencies are in your MATLAB path
        % method=1 (time-varying), time_skip=2.0, show=false (to avoid plots)
        % struct_opt is empty for default a0_type
        OUT_matlab = FluctuationStrength_Osses2016(signal', fs, 1, 2.0, false, []);
        
        % --- Print Key Results for Visual Comparison ---
        fprintf('\n--- MATLAB Results ---\n');
        
        fprintf('\nInstantaneous Fluctuation Strength (vacil):\n');
        fprintf('  First 5 values: %f, %f, %f, %f, %f\n', OUT_matlab.InstantaneousFluctuationStrength(1:5));
        fprintf('  Mean: %f\n', OUT_matlab.FSmean);
        fprintf('  Std:  %f\n', OUT_matlab.FSstd);
        fprintf('  Max:  %f\n', OUT_matlab.FSmax);
        fprintf('  Min:  %f\n', OUT_matlab.FSmin);
        
        fprintf('\nTime-Averaged Specific Fluctuation Strength (vacil/Bark):\n');
        fprintf('  First 5 values: %f, %f, %f, %f, %f\n', OUT_matlab.TimeAveragedSpecificFluctuationStrength(1:5));
        fprintf('  Mean: %f\n', mean(OUT_matlab.TimeAveragedSpecificFluctuationStrength));
        fprintf('  Std:  %f\n', std(OUT_matlab.TimeAveragedSpecificFluctuationStrength));
        fprintf('  Max:  %f\n', max(OUT_matlab.TimeAveragedSpecificFluctuationStrength));
        fprintf('  Min:  %f\n', min(OUT_matlab.TimeAveragedSpecificFluctuationStrength));
        
        fprintf('\nTime Vector (s):\n');
        fprintf('  First 5 values: %f, %f, %f, %f, %f\n', OUT_matlab.time(1:5));
        fprintf('  Length: %d\n', length(OUT_matlab.time));
        
        fprintf('\nBark Axis:\n');
        fprintf('  First 5 values: %f, %f, %f, %f, %f\n', OUT_matlab.barkAxis(1:5));
        fprintf('  Length: %d\n', length(OUT_matlab.barkAxis));
        
        fprintf('\n--- MATLAB Output Complete ---\n');

    %% SHARPNESS
    %-----------------------------------------------------------------------------------------------------------------------

    case "Sharpness_DIN45692"

        fs = 48000;         % Sampling frequency = 48 kHz
        fc = 1000.0;        % Center frequency = 1 kHz
        bw = 160;           % Bandwidth = 160 Hz
        Lp_dB = 60;         % Sound Pressure Level = 60 dB SPL
        duration = 5;       % Duration = 2 seconds
        
        num_samples = round(fs * duration); % Ensure integer number of samples
        white_noise = randn(num_samples, 1); % Column vector
        lowcut = fc - bw / 2;
        highcut = fc + bw / 2;
        nyquist = 0.5 * fs;
        low = lowcut / nyquist;
        high = highcut / nyquist;
        order = 4;
        [b, a] = butter(order, [low, high], 'band');
        narrowband_noise = filter(b, a, white_noise);
        P_ref = 20e-6; % Reference sound pressure in Pascals for 0 dB SPL
        P_rms_target = P_ref * (10^(Lp_dB / 20));
        current_rms = sqrt(mean(narrowband_noise.^2)); 
        if current_rms > 0
            calibrated_noise = narrowband_noise * (P_rms_target / current_rms);
        else
            calibrated_noise = zeros(size(narrowband_noise)); % Avoid division by zero
        end

        insig = calibrated_noise;
        
        P_ref = 20e-6;
        rms_value = sqrt(mean(insig.^2));
        measured_Lp_dB = 20 * log10(rms_value / P_ref);

        weight_type = 'DIN45692'; % 'DIN45692', 'bismarck', or 'aures'
        LoudnessField = 0;        % 0 for free field, 1 for diffuse field
        LoudnessMethod = 2;       % 1 for stationary, 2 for time-varying
        time_skip = 0.5;          % Skip first 0.5 seconds for statistics (if LoudnessMethod=2)
        show_sharpness = true;    % Display sharpness results
        show_loudness = false;    % Display loudness results
        
        OUT = Sharpness_DIN45692(insig, fs, weight_type, LoudnessField, ...
                                 LoudnessMethod, time_skip, show_sharpness, show_loudness);
        
        fprintf('\nSharpness Calculation Results (OUT):\n');
        if isfield(OUT, 'InstantaneousSharpness')
            fprintf('  Instantaneous Sharpness (first 5 values): ');
            fprintf('%.3f ', OUT.InstantaneousSharpness(1:min(5, end)));
            fprintf('\n');
            fprintf('  Mean Sharpness (Smean): %.3f acum\n', OUT.Smean);
            fprintf('  S5 Sharpness: %.3f acum\n', OUT.S5);
        elseif isfield(OUT, 'Sharpness')
            fprintf('  Stationary Sharpness: %.3f acum\n', OUT.Sharpness);
        else
            fprintf('  No sharpness output found (check LoudnessMethod).\n');
        end

    case "Sharpness_DIN45692_(loudness)"

        %% --- Test Case 1: Stationary Specific Loudness ---
        fprintf('\n--- Test Case 1: Stationary Sharpness ---\n');
        
        % Stationary specific loudness (1 time step, 24 Bark bands)
        stationary_specific_loudness = [
            0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ...
            1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 ...
            2.1 2.2 2.3 2.4 ];
        
        out_stationary = Sharpness_DIN45692_from_loudness( ...
            stationary_specific_loudness, ... % SpecificLoudness
            'DIN45692', ...                   % weight_type
            [], ...                           % time (unused)
            [], ...                           % time_skip
            false);                           % show_sharpness
        
        if isfield(out_stationary, 'Sharpness')
            fprintf('  Calculated Stationary Sharpness: %.3f acum\n', out_stationary.Sharpness);
        else
            warning('Stationary sharpness not found in output.');
        end
        
        %% --- Test Case 2: Time‑Varying Specific Loudness (Aures) ---
        fprintf('\n--- Test Case 2: Time‑Varying Sharpness (Aures) ---\n');
        
        num_time_steps = 50;
        num_bark_bands = 24;
        time_duration  = 5.0;                              % seconds
        time_vector    = linspace(0, time_duration, num_time_steps);
        
        % Generate time‑varying specific loudness
        rng default;                                       % reproducibility
        time_varying_specific_loudness = zeros(num_time_steps, num_bark_bands);
        for i = 1:num_time_steps
            factor = 1 + sin(time_vector(i) / time_duration * 2 * pi) * 0.5; % envelope
            time_varying_specific_loudness(i, :) = ...
                (rand(1, num_bark_bands) * 0.5 + 0.1) * factor;
        end
        % Ensure strictly positive values
        idx = time_varying_specific_loudness <= 0;
        time_varying_specific_loudness(idx) = 0.01;
        
        out_time_varying = Sharpness_DIN45692_from_loudness( ...
            time_varying_specific_loudness, ...
            'DIN45692', ...
            time_vector, ...     % time axis
            1.0, ...             % time_skip (s)
            false);               % show_sharpness
        
        if isfield(out_time_varying, 'InstantaneousSharpness')
            fprintf('  Instantaneous Sharpness (first 5 values):\n');
            disp(out_time_varying.InstantaneousSharpness(1:5)');
            fprintf('  Mean Sharpness (Smean): %.3f acum\n', out_time_varying.Smean(1));
            fprintf('  S5 Sharpness: %.3f acum\n', out_time_varying.S5(1));
            fprintf('  Time vector length: %d\n', numel(out_time_varying.time));
        else
            warning('Time‑varying sharpness (Aures) not found in output.');
        end

    %% PSYCHOACHOUSTIC ANNOYANCE (2016)
    %-----------------------------------------------------------------------------------------------------------------------

    case 'PsychoacousticAnnoyance_Di2016_PURE'
        fprintf('Generating Pure Sine Wave...\n');
        
        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = true;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 5.0;      % Duration (seconds)
        frequency = 1000;    % Sine wave frequency (Hz)
        amplitude = 0.1;     % Amplitude (Pa)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        insig = amplitude * sin(2 * pi * frequency * t);
        
        OUT_pure = PsychoacousticAnnoyance_Di2016(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_pure.PAmean);

    case 'PsychoacousticAnnoyance_Di2016_AM'
        fprintf('Generating Amplitude Modulated Sine Wave...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = true;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 5.0;      % Duration (seconds)
        carrier_freq = 1000; % Carrier frequency (Hz)
        mod_freq = 4;        % Modulation frequency (Hz)
        amplitude = 0.1;     % Base amplitude (Pa)
        mod_depth = 1.0;     % Modulation depth (0 to 1)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        
        carrier = sin(2 * pi * carrier_freq * t);
        % Modulator ensures amplitude varies from 0 to 1 (or 0 to 2*amplitude if not scaled by 0.5)
        modulator = 0.5 * (1 + mod_depth * sin(2 * pi * mod_freq * t));
        insig = amplitude * modulator .* carrier; % Element-wise multiplication
        
        OUT_am = PsychoacousticAnnoyance_Di2016(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_am);
        disp(OUT_am.PAmean);

    case 'PsychoacousticAnnoyance_Di2016_FM'
        fprintf('Generating Frequency Modulated Sine Wave...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 5.0;      % Duration (seconds)
        carrier_freq = 1000; % Carrier frequency (Hz)
        mod_freq = 70;       % Modulation frequency (Hz)
        freq_deviation = 100;% Frequency deviation (Hz)
        amplitude = 0.1;     % Amplitude (Pa)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        
        % Instantaneous phase for FM signal
        phase = 2 * pi * carrier_freq * t + (freq_deviation / mod_freq) * cos(2 * pi * mod_freq * t);
        insig = amplitude * sin(phase);
        
        OUT_fm = PsychoacousticAnnoyance_Di2016(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_fm);
        disp(OUT_fm.PAmean);

    case 'PsychoacousticAnnoyance_Di2016_NOISE'
        fprintf('Generating White Noise Signal...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 5.0;      % Duration (seconds)
        amplitude = 0.1;     % Amplitude scaling factor
        
        % Generate white noise. randn(1, N) creates a 1xN row vector of random numbers.
        insig = amplitude * randn(1, round(fs * duration));
        
        OUT_noise = PsychoacousticAnnoyance_Di2016(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_noise);
        disp(OUT_noise.PAmean);

    case 'PsychoacousticAnnoyance_Di2016_SHORT'
        fprintf('Generating Short Signal (less than 2 seconds)...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)
        
        fs = 48000;          % Sampling frequency (Hz)
        duration = 1.5;      % Duration (seconds) - IMPORTANT: less than 2s
        frequency = 1000;    % Sine wave frequency (Hz)
        amplitude = 0.1;     % Amplitude (Pa)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        insig = amplitude * sin(2 * pi * frequency * t);
        
        OUT_short = PsychoacousticAnnoyance_Di2016(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_short);

    case 'PsychoacousticAnnoyance_Di2016_PERCENT'
        fprintf('Generating Percentiles ...\n');

        N = 1;
        S = 1;
        R = 1;
        FS = 1;
        K = 1;

        OUT_per = PsychoacousticAnnoyance_Di2016_from_percentile(N,S,R,FS,K);
        disp(OUT_per)

    %% PSYCHOACHOUSTIC ANNOYANCE (1999)
    %-----------------------------------------------------------------------------------------------------------------------

    case 'PsychoacousticAnnoyance_Zwicker1999_PURE'
        fprintf('Generating Pure Sine Wave...\n');
        
        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 3.0;      % Duration (seconds)
        frequency = 1000;    % Sine wave frequency (Hz)
        amplitude = 0.1;     % Amplitude (Pa)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        insig = amplitude * sin(2 * pi * frequency * t);
        
        OUT_pure = PsychoacousticAnnoyance_Zwicker1999(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_pure.PAmean);

    case 'PsychoacousticAnnoyance_Zwicker1999_AM'
        fprintf('Generating Amplitude Modulated Sine Wave...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 3.0;      % Duration (seconds)
        carrier_freq = 1000; % Carrier frequency (Hz)
        mod_freq = 4;        % Modulation frequency (Hz)
        amplitude = 0.1;     % Base amplitude (Pa)
        mod_depth = 1.0;     % Modulation depth (0 to 1)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        
        carrier = sin(2 * pi * carrier_freq * t);
        % Modulator ensures amplitude varies from 0 to 1 (or 0 to 2*amplitude if not scaled by 0.5)
        modulator = 0.5 * (1 + mod_depth * sin(2 * pi * mod_freq * t));
        insig = amplitude * modulator .* carrier; % Element-wise multiplication
        
        OUT_am = PsychoacousticAnnoyance_Zwicker1999(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_am.PAmean);

    case 'PsychoacousticAnnoyance_Zwicker1999_FM'
        fprintf('Generating Frequency Modulated Sine Wave...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 3.0;      % Duration (seconds)
        carrier_freq = 1000; % Carrier frequency (Hz)
        mod_freq = 70;       % Modulation frequency (Hz)
        freq_deviation = 100;% Frequency deviation (Hz)
        amplitude = 0.1;     % Amplitude (Pa)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        
        % Instantaneous phase for FM signal
        phase = 2 * pi * carrier_freq * t + (freq_deviation / mod_freq) * cos(2 * pi * mod_freq * t);
        insig = amplitude * sin(phase);
        
        OUT_fm = PsychoacousticAnnoyance_Zwicker1999(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_fm.PAmean);

    case 'PsychoacousticAnnoyance_Zwicker1999_NOISE'
        fprintf('Generating White Noise Signal...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 3.0;      % Duration (seconds)
        amplitude = 0.1;     % Amplitude scaling factor
        
        % Generate white noise. randn(1, N) creates a 1xN row vector of random numbers.
        insig = amplitude * randn(1, round(fs * duration));
        
        OUT_noise = PsychoacousticAnnoyance_Zwicker1999(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_noise.PAmean);

    case 'PsychoacousticAnnoyance_Zwicker1999_SHORT'
        fprintf('Generating Short Signal (less than 2 seconds)...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)
        
        fs = 48000;          % Sampling frequency (Hz)
        duration = 1.5;      % Duration (seconds) - IMPORTANT: less than 2s
        frequency = 1000;    % Sine wave frequency (Hz)
        amplitude = 0.1;     % Amplitude (Pa)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        insig = amplitude * sin(2 * pi * frequency * t);
        
        OUT_short = PsychoacousticAnnoyance_Zwicker1999(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_short.ScalarPA);

    case 'PsychoacousticAnnoyance_Zwicker1999_PERCENT'
        fprintf('Generating Percentiles ...\n');

        N = 1;
        S = 1;
        R = 1;
        FS = 1;

        OUT_per = PsychoacousticAnnoyance_Zwicker1999_from_percentile(N,S,R,FS);
        disp(OUT_per)

    %% PSYCHOACHOUSTIC ANNOYANCE (2010)
    %-----------------------------------------------------------------------------------------------------------------------

    case 'PsychoacousticAnnoyance_More2010_PURE'
        fprintf('Generating Pure Sine Wave...\n');
        
        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 5.0;      % Duration (seconds)
        frequency = 1000;    % Sine wave frequency (Hz)
        amplitude = 0.1;     % Amplitude (Pa)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        insig = amplitude * sin(2 * pi * frequency * t);
        
        OUT_pure = PsychoacousticAnnoyance_More2010(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_pure.PAmean);

    case 'PsychoacousticAnnoyance_More2010_AM'
        fprintf('Generating Amplitude Modulated Sine Wave...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 5.0;      % Duration (seconds)
        carrier_freq = 1000; % Carrier frequency (Hz)
        mod_freq = 4;        % Modulation frequency (Hz)
        amplitude = 0.1;     % Base amplitude (Pa)
        mod_depth = 1.0;     % Modulation depth (0 to 1)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        
        carrier = sin(2 * pi * carrier_freq * t);
        % Modulator ensures amplitude varies from 0 to 1 (or 0 to 2*amplitude if not scaled by 0.5)
        modulator = 0.5 * (1 + mod_depth * sin(2 * pi * mod_freq * t));
        insig = amplitude * modulator .* carrier; % Element-wise multiplication
        
        OUT_am = PsychoacousticAnnoyance_More2010(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_am);
        disp(OUT_am.PAmean);

    case 'PsychoacousticAnnoyance_More2010_FM'
        fprintf('Generating Frequency Modulated Sine Wave...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 5.0;      % Duration (seconds)
        carrier_freq = 1000; % Carrier frequency (Hz)
        mod_freq = 70;       % Modulation frequency (Hz)
        freq_deviation = 100;% Frequency deviation (Hz)
        amplitude = 0.1;     % Amplitude (Pa)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        
        % Instantaneous phase for FM signal
        phase = 2 * pi * carrier_freq * t + (freq_deviation / mod_freq) * cos(2 * pi * mod_freq * t);
        insig = amplitude * sin(phase);
        
        OUT_fm = PsychoacousticAnnoyance_More2010(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_fm);
        disp(OUT_fm.PAmean);

    case 'PsychoacousticAnnoyance_More2010_NOISE'
        fprintf('Generating White Noise Signal...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)

        fs = 48000;          % Sampling frequency (Hz)
        duration = 5.0;      % Duration (seconds)
        amplitude = 0.1;     % Amplitude scaling factor
        
        % Generate white noise. randn(1, N) creates a 1xN row vector of random numbers.
        insig = amplitude * randn(1, round(fs * duration));
        
        OUT_noise = PsychoacousticAnnoyance_More2010(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_noise);
        disp(OUT_noise.PAmean);

    case 'PsychoacousticAnnoyance_More2010_SHORT'
        fprintf('Generating Short Signal (less than 2 seconds)...\n');

        LoudnessField = 0; % 0 = Free Field; 1 = Diffuse Field
        time_skip = 0.5;   % seconds to skip for statistics calculation
        showPA = false;     % Display psychoacoustic annoyance results
        show = false;       % Display individual metric results (Loudness, Sharpness, etc.)
        
        fs = 48000;          % Sampling frequency (Hz)
        duration = 1.5;      % Duration (seconds) - IMPORTANT: less than 2s
        frequency = 1000;    % Sine wave frequency (Hz)
        amplitude = 0.1;     % Amplitude (Pa)
        
        t = 0 : 1/fs : duration - 1/fs; % Time vector
        insig = amplitude * sin(2 * pi * frequency * t);
        
        OUT_short = PsychoacousticAnnoyance_More2010(insig, fs, LoudnessField, time_skip, showPA, show);
        disp(OUT_short);

    case 'PsychoacousticAnnoyance_More2010_PERCENT'
        fprintf('Generating Percentiles ...\n');

        N = 1;
        S = 1;
        R = 1;
        FS = 1;
        K = 1;

        OUT_per = PsychoacousticAnnoyance_More2010_from_percentile(N,S,R,FS,K);
        disp(OUT_per)

end
