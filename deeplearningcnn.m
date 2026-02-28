%% ================================================================
%  ECG ARRHYTHMIA DETECTION — FULL PIPELINE
%  Beat-level classification | SMOTE | 1D CNN vs ML Comparison
%  MIT-BIH Arrhythmia Database
%% ================================================================
clc; clear; close all;

records = {'105','106','107','108','109','111','112','113','114','115', ...
           '116','117','118','119','121','122','123','124','200','201', ...
           '202','203','205','207','208','209','210','212','213','214', ...
           '215','217','219','220','221','222','223','228','230','231', ...
           '232','233','234'};

%% -------- LABEL MAP (Record-level: 1=Arrhythmia, 0=Normal) --------
labelMap = containers.Map( ...
    {'105','106','107','108','109','111','112','113','114','115','116','117','118','119', ...
     '121','122','123','124','200','201','202','203','205','207','208','209','210','212', ...
     '213','214','215','217','219','220','221','222','223','228','230','231','232','233','234'}, ...
    [  1,    1,    1,    1,    1,    1,    0,    0,    1,    0,    1,    0,    1,    1, ...
       1,    0,    0,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    0, ...
       1,    1,    1,    1,    1,    0,    1,    0,    1,    1,    1,    1,    1,    1,    1]);

%% ================================================================
%  SECTION 1: BEAT SEGMENTATION + FEATURE EXTRACTION
%% ================================================================
% Beat window: 100 samples before R-peak, 200 samples after (at 360 Hz = ~0.83s)
WIN_PRE  = 100;
WIN_POST = 200;
WIN_LEN  = WIN_PRE + WIN_POST;  % 300 samples per beat

all_beats      = [];   % [N_beats x WIN_LEN] — raw beat segments for CNN
all_features   = [];   % [N_beats x 8]       — HRV+morphological for ML
all_beat_labels = [];  % [N_beats x 1]       — beat-level label

fprintf('=== PHASE 1: Beat Segmentation ===\n');

for i = 1:length(records)
    fprintf('Processing Record %s...\n', records{i});
    recLabel = labelMap(records{i});

    try
        [signal, Fs, ~] = rdsamp(records{i});
    catch
        fprintf('  [SKIP] Could not load record %s\n', records{i});
        continue;
    end

    ecg_raw = signal(:,1);

    %% -- Bandpass Filter --
    bpFilt = designfilt('bandpassiir', ...
        'FilterOrder',        4, ...
        'HalfPowerFrequency1', 0.5, ...
        'HalfPowerFrequency2', 40, ...
        'SampleRate',         Fs);
    ecg = filtfilt(bpFilt, ecg_raw);

    %% -- Normalize ECG to [-1, 1] --
    ecg = (ecg - min(ecg)) / (max(ecg) - min(ecg) + eps) * 2 - 1;

    %% -- R-Peak Detection --
    [~, locs] = findpeaks(ecg, 'MinPeakDistance', round(0.5*Fs), 'MinPeakHeight', 0.3);

    %% -- RR Intervals --
    RR = diff(locs) / Fs;
    RR = RR(RR > 0.3 & RR < 2.0);
    if length(RR) < 10, continue; end

    %% -- Beat Segmentation --
    valid_locs = locs(locs > WIN_PRE & locs <= length(ecg) - WIN_POST);

    for b = 1:length(valid_locs)
        r = valid_locs(b);
        beat = ecg(r - WIN_PRE : r + WIN_POST - 1)';  % 1 x WIN_LEN

        % Resample to standard 300 samples if Fs differs from 360
        if Fs ~= 360
            beat = resample(beat, 360, Fs);
            beat = beat(1:min(WIN_LEN, end));
            if length(beat) < WIN_LEN
                beat(end+1:WIN_LEN) = 0;
            end
        end

        all_beats = [all_beats; beat];
        all_beat_labels = [all_beat_labels; recLabel];
    end

    %% -- HRV + Morphological Features (per record, replicated per beat) --
    mean_RR = mean(RR);
    std_RR  = std(RR);
    rmssd   = sqrt(mean(diff(RR).^2));
    pnn50   = sum(abs(diff(RR)) > 0.05) / length(RR) * 100;
    cv_RR   = std_RR / (mean_RR + eps);

    % Frequency domain
    Fs_rr = 4;
    t_rr  = cumsum([0; RR(:)]);
    t_uni = (t_rr(1) : 1/Fs_rr : t_rr(end))';
    if length(t_uni) < 8, continue; end
    RR_uni = interp1(t_rr, [RR(1); RR(:)], t_uni, 'spline');
    [pxx, f] = pwelch(RR_uni - mean(RR_uni), [], [], [], Fs_rr);
    LF    = trapz(f(f>=0.04 & f<=0.15), pxx(f>=0.04 & f<=0.15));
    HF    = trapz(f(f>=0.15 & f<=0.40), pxx(f>=0.15 & f<=0.40));
    LF_HF = LF / (HF + eps);

    rec_feat = [mean_RR, std_RR, rmssd, pnn50, cv_RR, LF, HF, LF_HF];

    % Replicate features for each beat from this record
    n_beats_this = sum(valid_locs > WIN_PRE & valid_locs <= length(ecg) - WIN_POST);
    all_features = [all_features; repmat(rec_feat, n_beats_this, 1)];
end

fprintf('\nTotal beats extracted: %d\n', size(all_beats, 1));
fprintf('Arrhythmia beats: %d | Normal beats: %d\n', ...
    sum(all_beat_labels==1), sum(all_beat_labels==0));

%% ================================================================
%  SECTION 2: SMOTE — Synthetic Minority Oversampling
%% ================================================================
fprintf('\n=== PHASE 2: SMOTE Oversampling ===\n');

[all_features_bal, all_beat_labels_bal] = applySmote(all_features, all_beat_labels, 5);
[all_beats_bal,    ~]                   = applySmote(all_beats,     all_beat_labels, 5);

fprintf('After SMOTE — Arrhythmia: %d | Normal: %d\n', ...
    sum(all_beat_labels_bal==1), sum(all_beat_labels_bal==0));

%% ================================================================
%  SECTION 3: TRADITIONAL ML (SVM, KNN, Decision Tree) — 5-Fold CV
%% ================================================================
fprintf('\n=== PHASE 3: Traditional ML with 5-Fold CV ===\n');

features_norm = zscore(all_features_bal);
cv = cvpartition(all_beat_labels_bal, 'KFold', 5, 'Stratify', true);

modelNames = {'SVM (RBF)', 'KNN (k=5)', 'Decision Tree'};
mlMetrics  = initMetrics(modelNames);

for fold = 1:5
    Xtr = features_norm(training(cv,fold),:);
    Xte = features_norm(test(cv,fold),:);
    Ytr = all_beat_labels_bal(training(cv,fold));
    Yte = all_beat_labels_bal(test(cv,fold));

    % SVM
    mdl_svm  = fitcsvm(Xtr, Ytr, 'KernelFunction','rbf', 'BoxConstraint',1, 'Standardize',false);
    mlMetrics = evalModel(mlMetrics, 1, predict(mdl_svm,  Xte), Yte, fold);

    % KNN
    mdl_knn  = fitcknn(Xtr,  Ytr, 'NumNeighbors',5, 'Distance','euclidean');
    mlMetrics = evalModel(mlMetrics, 2, predict(mdl_knn,  Xte), Yte, fold);

    % Decision Tree
    mdl_tree = fitctree(Xtr, Ytr, 'MaxNumSplits',10);
    mlMetrics = evalModel(mlMetrics, 3, predict(mdl_tree, Xte), Yte, fold);
end

%% ================================================================
%  SECTION 4: 1D CNN (Deep Learning) — 5-Fold CV
%% ================================================================
fprintf('\n=== PHASE 4: 1D CNN Deep Learning ===\n');

cnnMetrics = initMetrics({'1D CNN'});
cv_cnn     = cvpartition(all_beat_labels_bal, 'KFold', 5, 'Stratify', true);

for fold = 1:5
    fprintf('  CNN Fold %d/5...\n', fold);

    Xtr_raw = all_beats_bal(training(cv_cnn,fold),:);
    Xte_raw = all_beats_bal(test(cv_cnn,fold),:);
    Ytr_raw = all_beat_labels_bal(training(cv_cnn,fold));
    Yte_raw = all_beat_labels_bal(test(cv_cnn,fold));

    % Reshape to [WIN_LEN x 1 x 1 x N] for MATLAB CNN input
    Xtr_4d = reshape(Xtr_raw', [WIN_LEN, 1, 1, size(Xtr_raw,1)]);
    Xte_4d = reshape(Xte_raw', [WIN_LEN, 1, 1, size(Xte_raw,1)]);

    Ytr_cat = categorical(Ytr_raw);
    Yte_cat = categorical(Yte_raw);

    %% -- 1D CNN Architecture --
    layers = [
        imageInputLayer([WIN_LEN 1 1], 'Normalization','zscore', 'Name','input')

        % Block 1
        convolution2dLayer([7 1], 32, 'Padding','same', 'Name','conv1')
        batchNormalizationLayer('Name','bn1')
        reluLayer('Name','relu1')
        maxPooling2dLayer([2 1], 'Stride',[2 1], 'Name','pool1')
        dropoutLayer(0.25, 'Name','drop1')

        % Block 2
        convolution2dLayer([5 1], 64, 'Padding','same', 'Name','conv2')
        batchNormalizationLayer('Name','bn2')
        reluLayer('Name','relu2')
        maxPooling2dLayer([2 1], 'Stride',[2 1], 'Name','pool2')
        dropoutLayer(0.25, 'Name','drop2')

        % Block 3
        convolution2dLayer([3 1], 128, 'Padding','same', 'Name','conv3')
        batchNormalizationLayer('Name','bn3')
        reluLayer('Name','relu3')
        maxPooling2dLayer([2 1], 'Stride',[2 1], 'Name','pool3')
        dropoutLayer(0.3, 'Name','drop3')

        % Fully Connected
        fullyConnectedLayer(128, 'Name','fc1')
        reluLayer('Name','relu_fc1')
        dropoutLayer(0.5, 'Name','drop_fc')
        fullyConnectedLayer(2,   'Name','fc2')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','output')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs',         20, ...
        'MiniBatchSize',     64, ...
        'InitialLearnRate',  1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 8, ...
        'L2Regularization',  1e-4, ...
        'ValidationData',    {Xte_4d, Yte_cat}, ...
        'ValidationFrequency', 30, ...
        'Shuffle',           'every-epoch', ...
        'Verbose',           false, ...
        'Plots',             'none');

    net      = trainNetwork(Xtr_4d, Ytr_cat, layers, options);
    y_pred   = classify(net, Xte_4d);
    cnnMetrics = evalModel(cnnMetrics, 1, double(string(y_pred)), Yte_raw, fold);
end

%% ================================================================
%  SECTION 5: RESULTS & COMPARISON
%% ================================================================
fprintf('\n========== FINAL PERFORMANCE COMPARISON ==========\n');
fprintf('%-18s %-10s %-10s %-10s %-10s\n', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score');
fprintf('%s\n', repmat('-', 1, 58));

allModelMetrics = [mlMetrics, cnnMetrics];
for m = 1:length(allModelMetrics)
    fprintf('%-18s %-10.4f %-10.4f %-10.4f %-10.4f\n', ...
        allModelMetrics(m).name, ...
        mean(allModelMetrics(m).acc),  ...
        mean(allModelMetrics(m).prec), ...
        mean(allModelMetrics(m).rec),  ...
        mean(allModelMetrics(m).f1));
end

fprintf('\nStandard Deviations (±):\n');
fprintf('%-18s %-10s %-10s %-10s %-10s\n', 'Model', 'Acc±', 'Prec±', 'Rec±', 'F1±');
fprintf('%s\n', repmat('-', 1, 58));
for m = 1:length(allModelMetrics)
    fprintf('%-18s %-10.4f %-10.4f %-10.4f %-10.4f\n', ...
        allModelMetrics(m).name, ...
        std(allModelMetrics(m).acc),  ...
        std(allModelMetrics(m).prec), ...
        std(allModelMetrics(m).rec),  ...
        std(allModelMetrics(m).f1));
end

%% -- Confusion Matrices (Aggregated) --
for m = 1:length(allModelMetrics)
    figure;
    confusionchart(allModelMetrics(m).cm, {'Arrhythmia','Normal'}, ...
        'Title',         ['Confusion Matrix — ' allModelMetrics(m).name ' (5-Fold Aggregated)'], ...
        'RowSummary',    'row-normalized', ...
        'ColumnSummary', 'column-normalized');
end

%% -- Bar Chart Comparison --
figure;
metricMeans = zeros(length(allModelMetrics), 4);
for m = 1:length(allModelMetrics)
    metricMeans(m,:) = [mean(allModelMetrics(m).acc),  mean(allModelMetrics(m).prec), ...
                        mean(allModelMetrics(m).rec),  mean(allModelMetrics(m).f1)];
end
bar(metricMeans);
set(gca, 'XTickLabel', {allModelMetrics.name});
legend({'Accuracy','Precision','Recall','F1-Score'}, 'Location','southoutside', 'Orientation','horizontal');
title('Model Performance Comparison (5-Fold CV)');
ylabel('Score');
ylim([0 1]);
grid on;

fprintf('\nDone.\n');

%% ================================================================
%  HELPER FUNCTIONS
%% ================================================================

function [X_out, Y_out] = applySmote(X, Y, k)
%APPLYSMOTE Synthetic Minority Oversampling Technique
%   Oversamples the minority class using k-nearest neighbor interpolation
    minority_idx = find(Y == 1);
    majority_idx = find(Y == 0);

    n_min = length(minority_idx);
    n_maj = length(majority_idx);

    if n_min >= n_maj
        X_out = X; Y_out = Y;
        return;
    end

    n_synth = n_maj - n_min;
    X_min   = X(minority_idx, :);
    synthetic = zeros(n_synth, size(X,2));

    for s = 1:n_synth
        % Pick random minority sample
        idx = randi(n_min);
        sample = X_min(idx,:);

        % Find k nearest neighbors within minority class
        dists = sum((X_min - sample).^2, 2);
        dists(idx) = Inf;
        [~, nn_idx] = sort(dists);
        nn_idx = nn_idx(1:min(k, end));

        % Interpolate
        neighbor = X_min(nn_idx(randi(length(nn_idx))), :);
        alpha    = rand(1, size(X,2));
        synthetic(s,:) = sample + alpha .* (neighbor - sample);
    end

    X_out = [X; synthetic];
    Y_out = [Y; ones(n_synth,1)];

    % Shuffle
    perm  = randperm(size(X_out,1));
    X_out = X_out(perm,:);
    Y_out = Y_out(perm);
end

function metrics = initMetrics(names)
%INITMETRICS Initialize metrics struct array
    metrics = struct();
    for i = 1:length(names)
        metrics(i).name = names{i};
        metrics(i).acc  = zeros(1,5);
        metrics(i).prec = zeros(1,5);
        metrics(i).rec  = zeros(1,5);
        metrics(i).f1   = zeros(1,5);
        metrics(i).cm   = zeros(2,2);
    end
end

function metrics = evalModel(metrics, idx, y_pred, y_true, fold)
%EVALMODEL Compute and store TP/TN/FP/FN metrics for one fold
    TP = sum(y_pred == 1 & y_true == 1);
    TN = sum(y_pred == 0 & y_true == 0);
    FP = sum(y_pred == 1 & y_true == 0);
    FN = sum(y_pred == 0 & y_true == 1);

    metrics(idx).acc(fold)  = (TP+TN) / (TP+TN+FP+FN+eps);
    metrics(idx).prec(fold) = TP / (TP+FP+eps);
    metrics(idx).rec(fold)  = TP / (TP+FN+eps);
    f1 = 2 * metrics(idx).prec(fold) * metrics(idx).rec(fold) / ...
             (metrics(idx).prec(fold) + metrics(idx).rec(fold) + eps);
    metrics(idx).f1(fold)   = f1;
    metrics(idx).cm         = metrics(idx).cm + [TP, FP; FN, TN];
end