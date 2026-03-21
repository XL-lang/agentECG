function summary = run_ecgdeli_analysis(inputCsv, fs, outputDir)
%RUN_ECGDELI_ANALYSIS Run ECGdeli on a CSV ECG matrix.
%
% inputCsv: CSV containing a numeric ECG matrix shaped samples x leads.
% fs: sampling rate in Hz.
% outputDir: directory for summary.json and results.mat.

if nargin ~= 3
    error('ECGdeliWrapper:InvalidArguments', ...
        'Expected inputCsv, fs, and outputDir arguments.');
end

fs = str2double(string(fs));
if isnan(fs) || fs <= 0
    error('ECGdeliWrapper:InvalidSamplingRate', ...
        'Sampling rate must be a positive number.');
end

inputCsv = char(inputCsv);
outputDir = char(outputDir);

if ~isfile(inputCsv)
    error('ECGdeliWrapper:MissingInput', 'Input CSV not found: %s', inputCsv);
end

if ~isfolder(outputDir)
    mkdir(outputDir);
end

summaryPath = fullfile(outputDir, 'summary.json');
resultsPath = fullfile(outputDir, 'results.mat');

try
    ecgdeliRoot = getenv('ECGDELI_ROOT');
    if isempty(ecgdeliRoot)
        scriptDir = fileparts(mfilename('fullpath'));
        repoRoot = fullfile(scriptDir, '..', '..', '..', '..');
        ecgdeliRoot = fullfile(repoRoot, 'external', 'ECGdeli');
    end
    ecgdeliRoot = char(java.io.File(ecgdeliRoot).getCanonicalPath());

    if ~isfolder(ecgdeliRoot)
        error('ECGdeliWrapper:MissingECGdeli', ...
            'ECGdeli root not found: %s', ecgdeliRoot);
    end

    addpath(genpath(ecgdeliRoot));

    ecg = readmatrix(inputCsv);
    if ~isnumeric(ecg) || ~ismatrix(ecg) || isempty(ecg) || size(ecg, 1) < 2 || size(ecg, 2) < 1
        error('ECGdeliWrapper:InvalidInput', ...
            'Input CSV must contain a numeric 2-D matrix shaped samples x leads.');
    end

    [ecgBaselineRemoved, ~] = ECG_Baseline_Removal(ecg, fs, 1, 0.5);
    ecgFiltered = ECG_High_Low_Filter(ecgBaselineRemoved, fs, 1, 40);
    ecgFiltered = Notch_Filter(ecgFiltered, fs, 50, 1);
    [ecgFiltered, ~, ~, ~] = Isoline_Correction(ecgFiltered);

    [fptMultiChannel, fptCell] = Annotate_ECG_Multi(ecgFiltered, fs);
    if isempty(fptMultiChannel) || isempty(fptCell)
        error('ECGdeliWrapper:NoFiducialPoints', ...
            'ECGdeli did not produce fiducial points.');
    end

    amplitudeFeatures = ExtractAmplitudeFeaturesFromFPT(fptCell, ecgFiltered);
    [timingFeatures, timingFeaturesSync] = ExtractIntervalFeaturesFromFPT(fptCell, fptMultiChannel);

    save(resultsPath, ...
        'fptMultiChannel', ...
        'fptCell', ...
        'amplitudeFeatures', ...
        'timingFeatures', ...
        'timingFeaturesSync', ...
        'fs', ...
        'inputCsv');

    summary = struct();
    summary.status = 'ok';
    summary.samples = size(ecg, 1);
    summary.leads = size(ecg, 2);
    summary.beats = size(fptMultiChannel, 1);
    summary.results_mat = resultsPath;
    summary.summary_json = summaryPath;
    summary.ecgdeli_root = ecgdeliRoot;
    write_json(summaryPath, summary);

    fprintf('ECGdeli analysis passed: %d samples, %d leads, %d beats.\n', ...
        summary.samples, summary.leads, summary.beats);
catch err
    summary = struct();
    summary.status = 'error';
    summary.error_identifier = err.identifier;
    summary.error_message = err.message;
    summary.summary_json = summaryPath;
    summary.results_mat = resultsPath;
    write_json(summaryPath, summary);
    rethrow(err);
end
end

function write_json(path, value)
fid = fopen(path, 'w');
if fid == -1
    error('ECGdeliWrapper:CannotWriteJson', 'Cannot write JSON: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(value, 'PrettyPrint', true));
end
