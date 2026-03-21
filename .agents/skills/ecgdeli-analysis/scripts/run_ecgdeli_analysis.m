function summary = run_ecgdeli_analysis(inputCsv, fs, outputDir)
%RUN_ECGDELI_ANALYSIS Run the ECGdeli analysis pipeline on a CSV ECG matrix.

if nargin ~= 3
    error('ECGDeliAnalysis:InvalidArguments', ...
        'Expected inputCsv, fs, and outputDir arguments.');
end

fs = str2double(string(fs));
if isnan(fs) || fs <= 0
    error('ECGDeliAnalysis:InvalidSamplingRate', ...
        'Sampling rate must be a positive number.');
end

inputCsv = char(inputCsv);
outputDir = char(outputDir);

if ~isfile(inputCsv)
    error('ECGDeliAnalysis:MissingInput', 'Input CSV not found: %s', inputCsv);
end

if ~isfolder(outputDir)
    mkdir(outputDir);
end

summaryPath = fullfile(outputDir, 'summary.json');
resultsPath = fullfile(outputDir, 'results.mat');
timingPath = fullfile(outputDir, 'timing_features.json');
amplitudePath = fullfile(outputDir, 'amplitude_features.json');

try
    ecgdeliRoot = getenv('ECGDELI_ROOT');
    if isempty(ecgdeliRoot)
        scriptDir = fileparts(mfilename('fullpath'));
        repoRoot = fullfile(scriptDir, '..', '..', '..', '..');
        ecgdeliRoot = fullfile(repoRoot, 'external', 'ECGdeli');
    end
    ecgdeliRoot = char(java.io.File(ecgdeliRoot).getCanonicalPath());

    if ~isfolder(ecgdeliRoot)
        error('ECGDeliAnalysis:MissingECGdeli', ...
            'ECGdeli root not found: %s', ecgdeliRoot);
    end

    addpath(genpath(ecgdeliRoot));

    ecg = readmatrix(inputCsv);
    if ~isnumeric(ecg) || ~ismatrix(ecg) || isempty(ecg) || size(ecg, 1) < 2 || size(ecg, 2) < 1
        error('ECGDeliAnalysis:InvalidInput', ...
            'Input CSV must contain a numeric 2-D matrix shaped samples x leads.');
    end

    [ecgBaselineRemoved, ~] = ECG_Baseline_Removal(ecg, fs, 1, 0.5);
    ecgFiltered = ECG_High_Low_Filter(ecgBaselineRemoved, fs, 1, 40);
    ecgFiltered = Notch_Filter(ecgFiltered, fs, 50, 1);
    [ecgFiltered, ~, ~, ~] = Isoline_Correction(ecgFiltered);

    [fptMultiChannel, fptCell] = Annotate_ECG_Multi(ecgFiltered, fs);
    if isempty(fptMultiChannel) || isempty(fptCell)
        error('ECGDeliAnalysis:NoFiducialPoints', ...
            'ECGdeli did not produce fiducial points.');
    end

    amplitudeFeatures = ExtractAmplitudeFeaturesFromFPT(fptCell, ecgFiltered);
    amplitudeFeatures = amplitudeFeatures(1:size(ecg, 2), :, :);
    [timingFeatures, timingFeaturesSync] = ExtractIntervalFeaturesFromFPT(fptCell, fptMultiChannel);

    save(resultsPath, ...
        'ecg', ...
        'ecgFiltered', ...
        'fptMultiChannel', ...
        'fptCell', ...
        'amplitudeFeatures', ...
        'timingFeatures', ...
        'timingFeaturesSync', ...
        'fs', ...
        'inputCsv');

    timingPayload = build_timing_payload(timingFeatures, timingFeaturesSync);
    amplitudePayload = build_amplitude_payload(amplitudeFeatures);
    write_json(timingPath, timingPayload);
    write_json(amplitudePath, amplitudePayload);

    summary = struct();
    summary.status = 'ok';
    summary.samples = size(ecg, 1);
    summary.leads = size(ecg, 2);
    summary.beats = size(fptMultiChannel, 1);
    summary.results_mat = resultsPath;
    summary.summary_json = summaryPath;
    summary.timing_features_json = timingPath;
    summary.amplitude_features_json = amplitudePath;
    summary.timing_features = feature_overview(timingPayload);
    summary.amplitude_features = feature_overview(amplitudePayload);
    summary.fiducial_points_available = true;
    summary.ecgdeli_root = ecgdeliRoot;
    write_json(summaryPath, summary);

    fprintf('ECGdeli analysis passed: %d samples, %d leads, %d beats.\n', ...
        summary.samples, summary.leads, summary.beats);
catch err
    summary = struct();
    summary.status = 'error';
    summary.samples = [];
    summary.leads = [];
    summary.beats = [];
    summary.results_mat = resultsPath;
    summary.summary_json = summaryPath;
    summary.timing_features_json = timingPath;
    summary.amplitude_features_json = amplitudePath;
    summary.timing_features = [];
    summary.amplitude_features = [];
    summary.fiducial_points_available = false;
    summary.error_identifier = err.identifier;
    summary.error_message = err.message;
    write_json(summaryPath, summary);
    rethrow(err);
end
end

function payload = build_timing_payload(leadwise, syncFeatures)
payload = struct();
payload.leadwise = struct();
payload.leadwise.feature_names = {'p_duration', 'qrs_duration', 't_duration', ...
    'pq_interval', 'pr_interval', 'qt_interval', 'rr_interval'};
payload.leadwise.shape = size(leadwise);
payload.leadwise.units = 'ms';
payload.leadwise.values = to_nested_3d(leadwise);

payload.synchronized = struct();
payload.synchronized.feature_names = {'p_duration', 'qrs_duration', 't_duration', ...
    'pq_interval', 'pr_interval', 'qt_interval', 'qtc_interval', 'rr_interval'};
payload.synchronized.shape = size(syncFeatures);
payload.synchronized.units = 'ms';
payload.synchronized.values = to_nested_2d(syncFeatures);
end

function payload = build_amplitude_payload(amplitudeFeatures)
payload = struct();
payload.feature_names = {'p_amplitude', 'q_amplitude', 'r_amplitude', 's_amplitude', 't_amplitude'};
payload.shape = size(amplitudeFeatures);
payload.units = 'signal_units';
payload.values = to_nested_3d(amplitudeFeatures);
end

function overview = feature_overview(payload)
overview = struct();
if isfield(payload, 'leadwise')
    overview.leadwise = small_feature_struct(payload.leadwise);
end
if isfield(payload, 'synchronized')
    overview.synchronized = small_feature_struct(payload.synchronized);
end
if isfield(payload, 'feature_names')
    overview = small_feature_struct(payload);
end
end

function payload = small_feature_struct(value)
payload = struct();
payload.feature_names = value.feature_names;
payload.shape = value.shape;
payload.units = value.units;
end

function rows = to_nested_2d(values)
[dim1, dim2] = size(values);
rows = cell(dim1, 1);
for i = 1:dim1
    rows{i} = reshape(num2cell(values(i, :)), 1, dim2);
end
end

function cubes = to_nested_3d(values)
[dim1, dim2, dim3] = size(values);
cubes = cell(dim1, 1);
for i = 1:dim1
    leadRows = cell(dim2, 1);
    for j = 1:dim2
        leadRows{j} = reshape(num2cell(squeeze(values(i, j, :))), 1, dim3);
    end
    cubes{i} = leadRows;
end
end

function write_json(path, value)
fid = fopen(path, 'w');
if fid == -1
    error('ECGDeliAnalysis:CannotWriteJson', 'Cannot write JSON: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(value, 'PrettyPrint', true));
end
