% run_sqat_matlab_full.m
% ======================
% Comprehensive MATLAB execution wrapper for ALL SQAT metrics.
% Runs every metric on three standard audio cases and exports results to
%   validation/matlab_results/results_matlab.json
%
% Metrics covered (12 total)
% --------------------------
%  1.  Loudness_ISO532_1               ISO 532-1:2017
%  2.  Roughness_Daniel1997            Daniel & Weber 1997
%  3.  Sharpness_DIN45692              DIN 45692:2009
%  4.  FluctuationStrength_Osses2016   Osses et al. 2016
%  5.  Tonality_Aures1985              Aures 1985
%  6.  Loudness_ECMA418_2              ECMA-418-2:2024 (Sottek HMS)
%  7.  Roughness_ECMA418_2             ECMA-418-2:2024 (Sottek HMS)
%  8.  Tonality_ECMA418_2              ECMA-418-2:2024 (Sottek HMS)
%  9.  PsychoacousticAnnoyance_Di2016
% 10.  PsychoacousticAnnoyance_More2010
% 11.  PsychoacousticAnnoyance_Zwicker1999
% 12.  EPNL_FAR_Part36
%
% Audio cases per metric
% ----------------------
%  reference   per-metric calibration signal (RefSignal_*.wav)
%              NOTE: PA metrics and EPNL have no dedicated reference signal
%  tester      tester.wav         (48 kHz, ~7.5 s, voice test signal)
%  a320        5sec_A320.wav      (44.1 kHz, 5 s, A320 take-off)
%
% Usage (in MATLAB)
% -----------------
%  cd '<SQAT4PY_ROOT>/original_matlab'
%  startup_SQAT
%  cd '<SQAT4PY_ROOT>'
%  run('validation/run_sqat_matlab_full.m')
%
% Output
% ------
%  validation/matlab_results/results_matlab.json
%
% Requirements
% ------------
%  MATLAB R2016b+               (jsonencode, local functions in scripts)
%  Signal Processing Toolbox    (all metrics)
%  Audio Toolbox                (ECMA-418-2 metrics only)
%
% Notes on calibration
% --------------------
%  SQAT convention: dBFS=94 means amplitude 1 = 1 Pa = 94 dBSPL.
%  gain = 10^((dBFS-94)/20)
%
%  RefSignal_Sharpness_DIN45692.wav is stored as int16 at a non-standard
%  level.  dBFS is auto-computed here so the RMS equals 60 dBSPL (1 acum).
%
% Notes on ECMA-418-2
% -------------------
%  Primary scalars per standard:
%    Loudness  : loudnessPowAvg  (power average, Section 8.1.4)
%    Roughness : roughness90Pc   (90th percentile = R10, Section 7.1.10)
%    Tonality  : tonalityAvg     (time average, Section 6.2.11)
%
% ============================================================================

clc;
fprintf('============================================================\n');
fprintf('  SQAT (MATLAB) -- Full Validation Wrapper\n');
fprintf('  %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('============================================================\n\n');

%% ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = fileparts(mfilename('fullpath'));   % .../SQAT4PY/validation
ROOT       = fileparts(SCRIPT_DIR);              % .../SQAT4PY
SF_DIR     = fullfile(ROOT, 'sound_files', 'reference_signals');
OUT_DIR    = fullfile(SCRIPT_DIR, 'matlab_results');
OUT_FILE   = fullfile(OUT_DIR, 'results_matlab.json');
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

%% ── Result container ─────────────────────────────────────────────────────────
res.implementation = 'matlab';
res.generated      = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
res.root           = ROOT;
res.dBFS_default   = 94.0;
res.results        = struct();

%% ── Shared metric parameters ─────────────────────────────────────────────────
show   = false;
showPA = false;

% Classic ISO / DIN / Daniel1997 metrics
L_field      = 0;          % free field
L_method     = 2;          % time-varying
L_time_skip  = 0.5;        % s
R_time_skip  = 0.2;        % s
S_weight     = 'DIN45692';
S_field      = 0;
S_method     = 2;
S_time_skip  = 0.5;        % s
FS_method    = 1;          % time-varying
FS_time_skip = 0.2;        % s
T_field      = 0;
T_time_skip  = 0.2;        % s

% ECMA-418-2 (Sottek Hearing Model)
ECMA_field      = 'free-frontal';
ECMA_time_skip  = 0.5;     % s  (minimum: 304 ms Loudness/Tonality, 320 ms Roughness)

% Psychoacoustic Annoyance
PA_LoudnessField = 0;
PA_time_skip     = 0.5;    % s

% EPNL (FAR Part 36)
EPNL_method    = 1;        % from calibrated audio signal
EPNL_dt        = 0.5;      % s  (FAR Part 36 standard interval)
EPNL_threshold = 10;       % TPNdB decay threshold for duration correction

%% ── Audio case tables ────────────────────────────────────────────────────────
% Each row: {case_name, filename, dBFS_or_NaN, target_SPL_or_NaN}
%   col 3 = NaN  : auto-compute dBFS from target SPL (col 4)
%   col 4 = NaN  : use dBFS directly from col 3

CASES_ISO  = {'reference', 'RefSignal_Loudness_ISO532_1.wav',               94,  NaN;
              'tester',    'tester.wav',                                     94,  NaN;
              'a320',      '5sec_A320.wav',                                  94,  NaN};

CASES_D97  = {'reference', 'RefSignal_Roughness_Daniel1997.wav',            94,  NaN;
              'tester',    'tester.wav',                                     94,  NaN;
              'a320',      '5sec_A320.wav',                                  94,  NaN};

CASES_SHP  = {'reference', 'RefSignal_Sharpness_DIN45692.wav',             NaN, 60.0;
              'tester',    'tester.wav',                                     94,  NaN;
              'a320',      '5sec_A320.wav',                                  94,  NaN};

CASES_FS   = {'reference', 'RefSignal_FluctuationStrength_Osses2016.wav',   94,  NaN;
              'tester',    'tester.wav',                                     94,  NaN;
              'a320',      '5sec_A320.wav',                                  94,  NaN};

CASES_T    = {'reference', 'RefSignal_Tonality_Aures1985.wav',              94,  NaN;
              'tester',    'tester.wav',                                     94,  NaN;
              'a320',      '5sec_A320.wav',                                  94,  NaN};

CASES_EL   = {'reference', 'RefSignal_Loudness_ECMA418_2.wav',              94,  NaN;
              'tester',    'tester.wav',                                     94,  NaN;
              'a320',      '5sec_A320.wav',                                  94,  NaN};

CASES_ER   = {'reference', 'RefSignal_Roughness_ECMA418_2.wav',             94,  NaN;
              'tester',    'tester.wav',                                     94,  NaN;
              'a320',      '5sec_A320.wav',                                  94,  NaN};

CASES_ET   = {'reference', 'RefSignal_Tonality_ECMA418_2.wav',              94,  NaN;
              'tester',    'tester.wav',                                     94,  NaN;
              'a320',      '5sec_A320.wav',                                  94,  NaN};

% PA and EPNL: no dedicated reference signal -> tester + a320 only
CASES_PA   = {'tester', 'tester.wav',    94, NaN;
              'a320',   '5sec_A320.wav', 94, NaN};

CASES_EPNL = {'tester', 'tester.wav',    94, NaN;
              'a320',   '5sec_A320.wav', 94, NaN};

%% ============================================================================
%%  1.  Loudness_ISO532_1   (ISO 532-1:2017, method 2, free field)
%% ============================================================================
metric = 'Loudness_ISO532_1';
fprintf('[%s]\n', metric);
res.results.(metric) = struct();
for ci = 1:size(CASES_ISO, 1)
    cname  = CASES_ISO{ci,1};  fname  = CASES_ISO{ci,2};
    dBFS_i = CASES_ISO{ci,3};  tSPL_i = CASES_ISO{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, struct('field',L_field,'method',L_method,'time_skip',L_time_skip));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = Loudness_ISO532_1(insig, fs, L_field, L_method, L_time_skip, show);
        e.scalars = struct( ...
            'Nmean', sc(OUT.Nmean), 'Nstd',  sc(OUT.Nstd),  ...
            'Nmax',  sc(OUT.Nmax),  'Nmin',  sc(OUT.Nmin),  ...
            'N5',    sc(OUT.N5),    'N10',   sc(OUT.N10),   ...
            'N50',   sc(OUT.N50),   'N95',   sc(OUT.N95),   ...
            'N_ratio', sc(OUT.N_ratio) );
        e.vectors = struct( ...
            'time',                  rv(OUT.time), ...
            'InstantaneousLoudness', rv(OUT.InstantaneousLoudness), ...
            'barkAxis',              rv(OUT.barkAxis), ...
            'SpecificLoudness_avg',  rv(mean(OUT.SpecificLoudness, 1)) );
        e.status = 'OK';
        fprintf('OK  Nmean=%.4f sone\n', sc(OUT.Nmean));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%%  2.  Roughness_Daniel1997
%% ============================================================================
metric = 'Roughness_Daniel1997';
fprintf('\n[%s]\n', metric);
res.results.(metric) = struct();
for ci = 1:size(CASES_D97, 1)
    cname  = CASES_D97{ci,1};  fname  = CASES_D97{ci,2};
    dBFS_i = CASES_D97{ci,3};  tSPL_i = CASES_D97{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, struct('time_skip', R_time_skip));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = Roughness_Daniel1997(insig, fs, R_time_skip, show);
        e.scalars = struct( ...
            'Rmean', sc(OUT.Rmean), 'Rstd', sc(OUT.Rstd), ...
            'Rmax',  sc(OUT.Rmax),  'Rmin', sc(OUT.Rmin), ...
            'R5',    sc(OUT.R5),    'R10',  sc(OUT.R10),  ...
            'R50',   sc(OUT.R50),   'R95',  sc(OUT.R95)   );
        e.vectors = struct( ...
            'time',                         rv(OUT.time), ...
            'InstantaneousRoughness',        rv(OUT.InstantaneousRoughness), ...
            'barkAxis',                     rv(OUT.barkAxis), ...
            'TimeAveragedSpecificRoughness', rv(OUT.TimeAveragedSpecificRoughness) );
        e.status = 'OK';
        fprintf('OK  Rmean=%.4f asper\n', sc(OUT.Rmean));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%%  3.  Sharpness_DIN45692   (DIN 45692 weighting, time-varying)
%%      NOTE: RefSignal stored as int16 at ~63.97 dBSPL full-scale.
%%            dBFS is auto-computed to reach exactly 60 dBSPL -> 1 acum.
%% ============================================================================
metric = 'Sharpness_DIN45692';
fprintf('\n[%s]\n', metric);
res.results.(metric) = struct();
for ci = 1:size(CASES_SHP, 1)
    cname  = CASES_SHP{ci,1};  fname  = CASES_SHP{ci,2};
    dBFS_i = CASES_SHP{ci,3};  tSPL_i = CASES_SHP{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, struct('weight_type',S_weight, ...
        'LoudnessField',S_field,'LoudnessMethod',S_method,'time_skip',S_time_skip));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = Sharpness_DIN45692(insig, fs, S_weight, S_field, S_method, ...
                                  S_time_skip, show, show);
        e.scalars = struct( ...
            'Smean', sc(OUT.Smean), 'Sstd', sc(OUT.Sstd), ...
            'Smax',  sc(OUT.Smax),  'Smin', sc(OUT.Smin), ...
            'S5',    sc(OUT.S5),    'S10',  sc(OUT.S10),  ...
            'S50',   sc(OUT.S50),   'S95',  sc(OUT.S95)   );
        e.vectors = struct( ...
            'time',                   rv(OUT.time), ...
            'InstantaneousSharpness', rv(OUT.InstantaneousSharpness) );
        e.status = 'OK';
        fprintf('OK  Smean=%.4f acum\n', sc(OUT.Smean));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%%  4.  FluctuationStrength_Osses2016   (method 1, time-varying)
%% ============================================================================
metric = 'FluctuationStrength_Osses2016';
fprintf('\n[%s]\n', metric);
res.results.(metric) = struct();
for ci = 1:size(CASES_FS, 1)
    cname  = CASES_FS{ci,1};  fname  = CASES_FS{ci,2};
    dBFS_i = CASES_FS{ci,3};  tSPL_i = CASES_FS{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, struct('method',FS_method,'time_skip',FS_time_skip));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = FluctuationStrength_Osses2016(insig, fs, FS_method, FS_time_skip, show);
        e.scalars = struct( ...
            'FSmean', sc(OUT.FSmean), 'FSstd', sc(OUT.FSstd), ...
            'FSmax',  sc(OUT.FSmax),  'FSmin', sc(OUT.FSmin), ...
            'FS5',    sc(OUT.FS5),    'FS10',  sc(OUT.FS10),  ...
            'FS50',   sc(OUT.FS50),   'FS95',  sc(OUT.FS95)   );
        e.vectors = struct( ...
            'time',                                    rv(OUT.time), ...
            'InstantaneousFluctuationStrength',         rv(OUT.InstantaneousFluctuationStrength), ...
            'barkAxis',                                rv(OUT.barkAxis), ...
            'TimeAveragedSpecificFluctuationStrength',  ...
                rv(OUT.TimeAveragedSpecificFluctuationStrength) );
        e.status = 'OK';
        fprintf('OK  FSmean=%.4f vacil\n', sc(OUT.FSmean));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%%  5.  Tonality_Aures1985   (free field)
%% ============================================================================
metric = 'Tonality_Aures1985';
fprintf('\n[%s]\n', metric);
res.results.(metric) = struct();
for ci = 1:size(CASES_T, 1)
    cname  = CASES_T{ci,1};  fname  = CASES_T{ci,2};
    dBFS_i = CASES_T{ci,3};  tSPL_i = CASES_T{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, struct('LoudnessField',T_field,'time_skip',T_time_skip));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = Tonality_Aures1985(insig, fs, T_field, T_time_skip, show);
        e.scalars = struct( ...
            'Kmean', sc(OUT.Kmean), 'Kstd', sc(OUT.Kstd), ...
            'Kmax',  sc(OUT.Kmax),  'Kmin', sc(OUT.Kmin), ...
            'K5',    sc(OUT.K5),    'K10',  sc(OUT.K10),  ...
            'K50',   sc(OUT.K50),   'K95',  sc(OUT.K95)   );
        e.vectors = struct( ...
            'time',                   rv(OUT.time), ...
            'InstantaneousTonality',  rv(OUT.InstantaneousTonality) );
        e.status = 'OK';
        fprintf('OK  Kmean=%.4f t.u.\n', sc(OUT.Kmean));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%%  6.  Loudness_ECMA418_2   (Sottek Hearing Model, free-frontal)
%%      Primary scalar: loudnessPowAvg  (ECMA-418-2:2024, Section 8.1.4)
%%      Reference: 1 kHz pure tone at 40 dBSPL -> 1 sone_HMS
%% ============================================================================
metric = 'Loudness_ECMA418_2';
fprintf('\n[%s]\n', metric);
res.results.(metric) = struct();
for ci = 1:size(CASES_EL, 1)
    cname  = CASES_EL{ci,1};  fname  = CASES_EL{ci,2};
    dBFS_i = CASES_EL{ci,3};  tSPL_i = CASES_EL{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, ...
        struct('fieldtype',ECMA_field,'time_skip',ECMA_time_skip));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = Loudness_ECMA418_2(insig, fs, ECMA_field, ECMA_time_skip, show);
        e.scalars = struct( ...
            'loudnessPowAvg', sc(OUT.loudnessPowAvg), ...
            'Nmean',  sc(OUT.Nmean),  'Nstd',  sc(OUT.Nstd),  ...
            'Nmax',   sc(OUT.Nmax),   'Nmin',  sc(OUT.Nmin),  ...
            'N5',     sc(OUT.N5),     'N10',   sc(OUT.N10),   ...
            'N50',    sc(OUT.N50),    'N95',   sc(OUT.N95)    );
        e.vectors = struct( ...
            'time',               rv(OUT.timeOut), ...
            'loudnessTDep',       rv(OUT.loudnessTDep), ...
            'bandCentreFreqs',    rv(OUT.bandCentreFreqs), ...
            'specLoudnessPowAvg', rv(OUT.specLoudnessPowAvg) );
        e.status = 'OK';
        fprintf('OK  loudnessPowAvg=%.4f sone_HMS\n', sc(OUT.loudnessPowAvg));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%%  7.  Roughness_ECMA418_2   (Sottek Hearing Model, free-frontal)
%%      Primary scalar: roughness90Pc = R10  (ECMA-418-2:2024, Section 7.1.10)
%%      Reference: 1 kHz tone 100% AM at 70 Hz at 60 dBSPL -> 1 asper
%% ============================================================================
metric = 'Roughness_ECMA418_2';
fprintf('\n[%s]\n', metric);
res.results.(metric) = struct();
for ci = 1:size(CASES_ER, 1)
    cname  = CASES_ER{ci,1};  fname  = CASES_ER{ci,2};
    dBFS_i = CASES_ER{ci,3};  tSPL_i = CASES_ER{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, ...
        struct('fieldtype',ECMA_field,'time_skip',ECMA_time_skip));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = Roughness_ECMA418_2(insig, fs, ECMA_field, ECMA_time_skip, show);
        % R10 = value exceeded 10 % of time = 90th percentile = roughness90Pc
        e.scalars = struct( ...
            'roughness90Pc', sc(OUT.roughness90Pc), ...
            'Rmean', sc(OUT.Rmean), 'Rstd',  sc(OUT.Rstd),  ...
            'Rmax',  sc(OUT.Rmax),  'Rmin',  sc(OUT.Rmin),  ...
            'R5',    sc(OUT.R5),    'R10',   sc(OUT.R10),   ...
            'R50',   sc(OUT.R50),   'R90',   sc(OUT.R90)    );
        e.vectors = struct( ...
            'time',              rv(OUT.timeOut), ...
            'roughnessTDep',     rv(OUT.roughnessTDep), ...
            'bandCentreFreqs',   rv(OUT.bandCentreFreqs), ...
            'specRoughnessAvg',  rv(OUT.specRoughnessAvg) );
        e.status = 'OK';
        fprintf('OK  roughness90Pc=%.4f asper\n', sc(OUT.roughness90Pc));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%%  8.  Tonality_ECMA418_2   (Sottek Hearing Model, free-frontal)
%%      Primary scalar: tonalityAvg  (ECMA-418-2:2024, Section 6.2.11)
%%      Reference: 1 kHz pure tone at 40 dBSPL -> 1 tu_HMS
%% ============================================================================
metric = 'Tonality_ECMA418_2';
fprintf('\n[%s]\n', metric);
res.results.(metric) = struct();
for ci = 1:size(CASES_ET, 1)
    cname  = CASES_ET{ci,1};  fname  = CASES_ET{ci,2};
    dBFS_i = CASES_ET{ci,3};  tSPL_i = CASES_ET{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, ...
        struct('fieldtype',ECMA_field,'time_skip',ECMA_time_skip));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = Tonality_ECMA418_2(insig, fs, ECMA_field, ECMA_time_skip, show);
        e.scalars = struct( ...
            'tonalityAvg', sc(OUT.tonalityAvg), ...
            'Tmean', sc(OUT.Tmean), 'Tstd',  sc(OUT.Tstd),  ...
            'Tmax',  sc(OUT.Tmax),  'Tmin',  sc(OUT.Tmin),  ...
            'T5',    sc(OUT.T5),    'T10',   sc(OUT.T10),   ...
            'T50',   sc(OUT.T50),   'T95',   sc(OUT.T95)    );
        e.vectors = struct( ...
            'time',              rv(OUT.timeOut), ...
            'tonalityTDep',      rv(OUT.tonalityTDep), ...
            'bandCentreFreqs',   rv(OUT.bandCentreFreqs), ...
            'specTonalityAvg',   rv(OUT.specTonalityAvg) );
        e.status = 'OK';
        fprintf('OK  tonalityAvg=%.4f tu_HMS\n', sc(OUT.tonalityAvg));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%%  9.  PsychoacousticAnnoyance_Di2016   (5 sub-metrics: N, S, R, FS, K)
%%      No dedicated reference signal.
%% ============================================================================
metric = 'PsychoacousticAnnoyance_Di2016';
fprintf('\n[%s]\n', metric);
res.results.(metric) = struct();
for ci = 1:size(CASES_PA, 1)
    cname  = CASES_PA{ci,1};  fname  = CASES_PA{ci,2};
    dBFS_i = CASES_PA{ci,3};  tSPL_i = CASES_PA{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, ...
        struct('LoudnessField',PA_LoudnessField,'time_skip',PA_time_skip));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = PsychoacousticAnnoyance_Di2016(insig, fs, PA_LoudnessField, ...
                                              PA_time_skip, showPA, show);
        e.scalars = struct( ...
            'PAmean',   sc(OUT.PAmean),   'PAstd',  sc(OUT.PAstd),  ...
            'PAmax',    sc(OUT.PAmax),    'PAmin',  sc(OUT.PAmin),  ...
            'PA5',      sc(OUT.PA5),      'PA10',   sc(OUT.PA10),   ...
            'PA50',     sc(OUT.PA50),     'PA95',   sc(OUT.PA95),   ...
            'ScalarPA', sc(OUT.ScalarPA), ...
            'Nmean',    sc(OUT.L.Nmean),  'Smean',  sc(OUT.S.Smean), ...
            'Rmean',    sc(OUT.R.Rmean),  'FSmean', sc(OUT.FS.FSmean), ...
            'Kmean',    sc(OUT.K.Kmean)  );
        e.vectors = struct( ...
            'time',             rv(OUT.time), ...
            'InstantaneousPA',  rv(OUT.InstantaneousPA), ...
            'wt',               rv(OUT.wt), ...    % tonality+loudness weighting
            'wfr',              rv(OUT.wfr), ...   % fluct.strength+roughness weighting
            'ws',               rv(OUT.ws)  );     % sharpness+loudness weighting
        e.status = 'OK';
        fprintf('OK  PAmean=%.4f\n', sc(OUT.PAmean));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%% 10.  PsychoacousticAnnoyance_More2010   (5 sub-metrics: N, S, R, FS, K)
%%      No dedicated reference signal.
%% ============================================================================
metric = 'PsychoacousticAnnoyance_More2010';
fprintf('\n[%s]\n', metric);
res.results.(metric) = struct();
for ci = 1:size(CASES_PA, 1)
    cname  = CASES_PA{ci,1};  fname  = CASES_PA{ci,2};
    dBFS_i = CASES_PA{ci,3};  tSPL_i = CASES_PA{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, ...
        struct('LoudnessField',PA_LoudnessField,'time_skip',PA_time_skip));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = PsychoacousticAnnoyance_More2010(insig, fs, PA_LoudnessField, ...
                                                PA_time_skip, showPA, show);
        e.scalars = struct( ...
            'PAmean',   sc(OUT.PAmean),   'PAstd',  sc(OUT.PAstd),  ...
            'PAmax',    sc(OUT.PAmax),    'PAmin',  sc(OUT.PAmin),  ...
            'PA5',      sc(OUT.PA5),      'PA10',   sc(OUT.PA10),   ...
            'PA50',     sc(OUT.PA50),     'PA95',   sc(OUT.PA95),   ...
            'ScalarPA', sc(OUT.ScalarPA), ...
            'Nmean',    sc(OUT.L.Nmean),  'Smean',  sc(OUT.S.Smean), ...
            'Rmean',    sc(OUT.R.Rmean),  'FSmean', sc(OUT.FS.FSmean), ...
            'Kmean',    sc(OUT.K.Kmean)  );
        e.vectors = struct( ...
            'time',             rv(OUT.time), ...
            'InstantaneousPA',  rv(OUT.InstantaneousPA), ...
            'wt',               rv(OUT.wt), ...
            'wfr',              rv(OUT.wfr), ...
            'ws',               rv(OUT.ws)  );
        e.status = 'OK';
        fprintf('OK  PAmean=%.4f\n', sc(OUT.PAmean));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%% 11.  PsychoacousticAnnoyance_Zwicker1999   (4 sub-metrics: N, S, R, FS)
%%      NOTE: Zwicker 1999 does NOT include tonality (no wt, no K sub-metric).
%%      No dedicated reference signal.
%% ============================================================================
metric = 'PsychoacousticAnnoyance_Zwicker1999';
fprintf('\n[%s]\n', metric);
res.results.(metric) = struct();
for ci = 1:size(CASES_PA, 1)
    cname  = CASES_PA{ci,1};  fname  = CASES_PA{ci,2};
    dBFS_i = CASES_PA{ci,3};  tSPL_i = CASES_PA{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, ...
        struct('LoudnessField',PA_LoudnessField,'time_skip',PA_time_skip));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = PsychoacousticAnnoyance_Zwicker1999(insig, fs, PA_LoudnessField, ...
                                                   PA_time_skip, showPA, show);
        e.scalars = struct( ...
            'PAmean',   sc(OUT.PAmean),   'PAstd',  sc(OUT.PAstd),  ...
            'PAmax',    sc(OUT.PAmax),    'PAmin',  sc(OUT.PAmin),  ...
            'PA5',      sc(OUT.PA5),      'PA10',   sc(OUT.PA10),   ...
            'PA50',     sc(OUT.PA50),     'PA95',   sc(OUT.PA95),   ...
            'ScalarPA', sc(OUT.ScalarPA), ...
            'Nmean',    sc(OUT.L.Nmean),  'Smean',  sc(OUT.S.Smean), ...
            'Rmean',    sc(OUT.R.Rmean),  'FSmean', sc(OUT.FS.FSmean) );
        % Zwicker 1999 has no tonality -> no wt field
        e.vectors = struct( ...
            'time',             rv(OUT.time), ...
            'InstantaneousPA',  rv(OUT.InstantaneousPA), ...
            'wfr',              rv(OUT.wfr), ...
            'ws',               rv(OUT.ws)  );
        e.status = 'OK';
        fprintf('OK  PAmean=%.4f\n', sc(OUT.PAmean));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%% 12.  EPNL_FAR_Part36   (method=1: from audio signal, dt=0.5 s)
%%      NOTE: EPNL is designed for aircraft flyover certification noise.
%%            Results for non-aviation signals are not physically meaningful.
%%            No dedicated reference signal -> tester + a320 only.
%% ============================================================================
metric = 'EPNL_FAR_Part36';
fprintf('\n[%s]\n', metric);
fprintf('  NOTE: EPNL is designed for aircraft flyover noise.\n');
fprintf('        tester.wav result is not physically meaningful.\n');
res.results.(metric) = struct();
for ci = 1:size(CASES_EPNL, 1)
    cname  = CASES_EPNL{ci,1};  fname  = CASES_EPNL{ci,2};
    dBFS_i = CASES_EPNL{ci,3};  tSPL_i = CASES_EPNL{ci,4};
    fprintf('  %-10s (%s) ... ', cname, fname);
    e = mk_entry(fname, dBFS_i, struct('method',EPNL_method,'dt',EPNL_dt, ...
        'threshold',EPNL_threshold));
    try
        [insig, fs, deff] = load_sig(SF_DIR, fname, dBFS_i, tSPL_i);
        e.dBFS = deff;  e.fs = double(fs);
        OUT = EPNL_FAR_Part36(insig, fs, EPNL_method, EPNL_dt, EPNL_threshold, show);
        e.scalars = struct( ...
            'EPNL',   sc(OUT.EPNL),   ...  % Effective Perceived Noise Level (EPNdB)
            'PNLTM',  sc(OUT.PNLTM),  ...  % max Tone-Corrected PNL (TPNdB)
            'PNLM',   sc(OUT.PNLM)    );   % max Perceived Noise Level (PNdB)
        e.vectors = struct( ...
            'time',        rv(OUT.time),  ...  % dt-averaged time vector
            'PNL',         rv(OUT.PNL),   ...  % Perceived Noise Level vs time
            'PNLT',        rv(OUT.PNLT),  ...  % Tone-Corrected PNL vs time
            'TOB_freq',    rv(OUT.TOB_freq), ...              % 24 third-octave centre freqs
            'SPL_TOB_avg', rv(mean(OUT.SPL_TOB_spectra, 1)) ); % time-avg TOB spectrum
        e.status = 'OK';
        fprintf('OK  EPNL=%.4f EPNdB\n', sc(OUT.EPNL));
    catch ME
        e.error = ME.message;  fprintf('FAIL: %s\n', ME.message);
    end
    res.results.(metric).(cname) = e;
end

%% ============================================================================
%%  JSON export
%% ============================================================================
fprintf('\n[Writing JSON] %s\n', OUT_FILE);
json_str = jsonencode(res, 'PrettyPrint', true);
fid = fopen(OUT_FILE, 'w', 'n', 'UTF-8');
fprintf(fid, '%s\n', json_str);
fclose(fid);
fprintf('[OK] Results written -> %s\n', OUT_FILE);

%% ============================================================================
%%  Console summary
%% ============================================================================
metrics_all = fieldnames(res.results);
fprintf('\n============================================================\n');
fprintf('  SUMMARY\n');
fprintf('------------------------------------------------------------\n');
for mi = 1:numel(metrics_all)
    m  = metrics_all{mi};
    cs = fieldnames(res.results.(m));
    for ci = 1:numel(cs)
        c  = cs{ci};
        st = res.results.(m).(c).status;
        fprintf('  %-40s %-10s  %s\n', m, c, st);
    end
end
fprintf('============================================================\n');

% ============================================================================
%   LOCAL FUNCTIONS   (must appear after all script code in R2016b+)
% ============================================================================

function [insig, fs, dBFS_eff] = load_sig(sf_dir, fname, dBFS_in, tSPL)
% Load WAV, convert to mono double, apply SQAT dBFS calibration.
% If tSPL is not NaN, auto-compute dBFS so that RMS equals tSPL dBSPL.
    fpath = fullfile(sf_dir, fname);
    [raw, fs] = audioread(fpath);
    if size(raw, 2) > 1, raw = mean(raw, 2); end   % stereo -> mono
    raw = double(raw);
    if ~isnan(tSPL)
        rms_raw  = sqrt(mean(raw .^ 2));
        p_target = 2e-5 * 10 ^ (tSPL / 20);
        dBFS_in  = 94 + 20 * log10(p_target / rms_raw);
    end
    dBFS_eff = double(dBFS_in);
    insig    = 10 ^ ((dBFS_eff - 94) / 20) * raw;
end

function e = mk_entry(fname, dBFS_v, params)
% Create a blank result entry struct.
    e = struct( ...
        'audio_file', fname,   'dBFS',       [],  'fs', [], ...
        'parameters', params,  'status',     'ERROR', ...
        'error',      '',      'scalars',    struct(), ...
        'vectors',    struct() );
    if ~isnan(dBFS_v), e.dBFS = double(dBFS_v); end
end

function s = sc(x)
% Safe scalar: return first element as real double.
    s = real(double(x(1)));
end

function v = rv(x)
% Row vector: flatten any array to a real double 1-D row.
% real() strips any tiny imaginary parts that may arise from FFT-based metrics.
    v = real(double(x(:)))';
end
