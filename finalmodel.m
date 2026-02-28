clc;
clear;
close all;

records = {'105', '106', '107', '108', '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', ...
           '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', ...
           '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'};

%% -------- LABEL ASSIGNMENT (MIT-BIH Arrhythmia Database) --------
% 1 = Arrhythmia, 0 = Normal
% Based on standard MIT-BIH annotations
labelMap = containers.Map( ...
    {'105','106','107','108','109','111','112','113','114','115','116','117','118','119', ...
     '121','122','123','124','200','201','202','203','205','207','208','209','210','212', ...
     '213','214','215','217','219','220','221','222','223','228','230','231','232','233','234'}, ...
    [  1,    1,    1,    1,    1,    1,    0,    0,    1,    0,    1,    0,    1,    1, ...
       1,    0,    0,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    0, ...
       1,    1,    1,    1,    1,    0,    1,    0,    1,    1,    1,    1,    1,    1,    1]);

features = [];
labels   = [];

for i = 1:length(records)
    fprintf('\nProcessing Record %s\n', records{i});

    [signal, Fs, tm] = rdsamp(records{i});
    ecg_raw = signal(:,1);

    %% -------- BANDPASS FILTER --------
    bpFilt = designfilt('bandpassiir', ...
        'FilterOrder', 4, ...
        'HalfPowerFrequency1', 0.5, ...
        'HalfPowerFrequency2', 40, ...
        'SampleRate', Fs);
    ecg = filtfilt(bpFilt, ecg_raw);

    %% -------- R-PEAK DETECTION --------
    [~, locs] = findpeaks(ecg, 'MinPeakDistance', 0.6*Fs);

    %% -------- RR INTERVAL ANALYSIS --------
    RR = diff(locs) / Fs;
    RR = RR(RR > 0.3 & RR < 2);

    if length(RR) < 5
        fprintf('  Skipping record %s — insufficient RR intervals.\n', records{i});
        continue;
    end

    %% -------- HRV TIME DOMAIN FEATURES --------
    mean_RR  = mean(RR);
    std_RR   = std(RR);
    rmssd    = sqrt(mean(diff(RR).^2));
    pnn50    = sum(abs(diff(RR)) > 0.05) / length(diff(RR)) * 100;  % pNN50 (%)
    cv_RR    = std_RR / mean_RR;                                     % Coefficient of Variation

    %% -------- HRV FREQUENCY DOMAIN FEATURES --------
    Fs_rr = 4;  % Resample RR to uniform 4 Hz grid
    t_rr  = cumsum([0; RR(:)]);
    t_uni = (t_rr(1):1/Fs_rr:t_rr(end))';
    RR_uni = interp1(t_rr, [RR(1); RR(:)], t_uni, 'spline');

    [pxx, f] = pwelch(RR_uni - mean(RR_uni), [], [], [], Fs_rr);

    lf_band = f >= 0.04 & f <= 0.15;
    hf_band = f >= 0.15 & f <= 0.40;

    LF  = trapz(f(lf_band), pxx(lf_band));
    HF  = trapz(f(hf_band), pxx(hf_band));
    LF_HF = LF / (HF + eps);

    %% -------- STORE FEATURES & LABELS --------
    features = [features; mean_RR, std_RR, rmssd, pnn50, cv_RR, LF, HF, LF_HF];
    labels   = [labels;   labelMap(records{i})];
end

fprintf('\nTotal records processed: %d\n', length(labels));
fprintf('Arrhythmia (1): %d | Normal (0): %d\n', sum(labels==1), sum(labels==0));

%% -------- FEATURE NORMALIZATION (Z-score) --------
[features_norm, mu, sigma] = zscore(features);

%% -------- 5-FOLD CROSS VALIDATION SETUP --------
cv = cvpartition(labels, 'KFold', 5, 'Stratify', true);

models    = {'SVM', 'KNN', 'DecisionTree'};
numModels = length(models);

% Storage for per-fold metrics
allMetrics = struct();
for m = 1:numModels
    allMetrics(m).name = models{m};
    allMetrics(m).acc  = zeros(1,5);
    allMetrics(m).prec = zeros(1,5);
    allMetrics(m).rec  = zeros(1,5);
    allMetrics(m).f1   = zeros(1,5);
    allMetrics(m).cm   = zeros(2,2);
end

%% -------- TRAINING & EVALUATION --------
for fold = 1:5
    trainIdx = training(cv, fold);
    testIdx  = test(cv, fold);

    X_train = features_norm(trainIdx, :);
    X_test  = features_norm(testIdx,  :);
    y_train = labels(trainIdx);
    y_test  = labels(testIdx);

    for m = 1:numModels
        switch models{m}
            case 'SVM'
                mdl = fitcsvm(X_train, y_train, ...
                    'KernelFunction', 'rbf', ...
                    'BoxConstraint',  1, ...
                    'Standardize',    false);

            case 'KNN'
                mdl = fitcknn(X_train, y_train, ...
                    'NumNeighbors', 5, ...
                    'Distance',     'euclidean');

            case 'DecisionTree'
                mdl = fitctree(X_train, y_train, ...
                    'MaxNumSplits', 10);
        end

        y_pred = predict(mdl, X_test);

        % Confusion matrix components
        TP = sum(y_pred == 1 & y_test == 1);
        TN = sum(y_pred == 0 & y_test == 0);
        FP = sum(y_pred == 1 & y_test == 0);
        FN = sum(y_pred == 0 & y_test == 1);

        acc  = (TP + TN) / (TP + TN + FP + FN);
        prec = TP / (TP + FP + eps);
        rec  = TP / (TP + FN + eps);
        f1   = 2 * prec * rec / (prec + rec + eps);

        allMetrics(m).acc(fold)  = acc;
        allMetrics(m).prec(fold) = prec;
        allMetrics(m).rec(fold)  = rec;
        allMetrics(m).f1(fold)   = f1;
        allMetrics(m).cm         = allMetrics(m).cm + [TP, FP; FN, TN];
    end
end

%% -------- PERFORMANCE SUMMARY --------
fprintf('\n========== CROSS-VALIDATION PERFORMANCE SUMMARY ==========\n');
fprintf('%-15s %-12s %-12s %-12s %-12s\n', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score');
fprintf('%s\n', repmat('-', 1, 63));

for m = 1:numModels
    fprintf('%-15s %-12.4f %-12.4f %-12.4f %-12.4f\n', ...
        allMetrics(m).name, ...
        mean(allMetrics(m).acc), ...
        mean(allMetrics(m).prec), ...
        mean(allMetrics(m).rec), ...
        mean(allMetrics(m).f1));
end

%% -------- CONFUSION MATRICES --------
for m = 1:numModels
    figure;
    cm_data = allMetrics(m).cm;
    confusionchart(cm_data, {'Arrhythmia','Normal'}, ...
        'Title',         ['Confusion Matrix — ' allMetrics(m).name ' (Aggregated 5-Fold)'], ...
        'RowSummary',    'row-normalized', ...
        'ColumnSummary', 'column-normalized');
end