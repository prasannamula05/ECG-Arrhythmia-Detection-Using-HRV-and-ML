clc;
clear;
close all;

records = {'105', '106', '107', '108', '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', '122',
    '123', '124', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '217', '219',
    '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'};
features = [];

for i = 1:length(records)
    
    fprintf('\nProcessing Record %s\n', records{i});
    
    [signal, Fs, tm] = rdsamp(records{i});
    
    ecg_raw = signal(:,1);
    
    %% -------- 1. RAW ECG PLOT --------
    figure;
    plot(tm(1:3000), ecg_raw(1:3000));
    title(['Raw ECG - Record ' records{i}]);
    xlabel('Time (s)');
    ylabel('Amplitude');
    
    %% -------- 2. BANDPASS FILTER --------
    bpFilt = designfilt('bandpassiir', ...
        'FilterOrder', 4, ...
        'HalfPowerFrequency1', 0.5, ...
        'HalfPowerFrequency2', 40, ...
        'SampleRate', Fs);
    
    ecg = filtfilt(bpFilt, ecg_raw);
    
    figure;
    plot(tm(1:3000), ecg(1:3000));
    title(['Filtered ECG - Record ' records{i}]);
    xlabel('Time (s)');
    ylabel('Amplitude');
    
    %% -------- 3. R-PEAK DETECTION --------
    [pks, locs] = findpeaks(ecg,'MinPeakDistance',0.6*Fs);
    
    figure;
    plot(tm(1:3000), ecg(1:3000));
    hold on;
    plot(tm(locs), pks, 'ro');
    title(['R Peaks - Record ' records{i}]);
    xlabel('Time (s)');
    ylabel('Amplitude');
    
    %% -------- 4. RR INTERVAL ANALYSIS --------
    RR = diff(locs)/Fs;
    RR = RR(RR > 0.3 & RR < 2);
    
    figure;
    plot(RR);
    title(['RR Intervals - Record ' records{i}]);
    xlabel('Beat Number');
    ylabel('RR Interval (s)');
    
    %% -------- 5. HRV TIME DOMAIN FEATURES --------
    mean_RR = mean(RR);
    std_RR = std(RR);
    rmssd = sqrt(mean(diff(RR).^2));
    
    %% -------- 6. HRV FREQUENCY DOMAIN --------
    Fs_rr = 1/mean_RR;
    
    [pxx,f] = pwelch(RR - mean(RR),[],[],[],Fs_rr);
    
    figure;
    plot(f,10*log10(pxx));
    title(['HRV Power Spectrum - Record ' records{i}]);
    xlabel('Frequency (Hz)');
    ylabel('Power (dB)');
    
    %% -------- STORE FEATURES --------
    features = [features; mean_RR std_RR rmssd];
    
end

disp('Feature Matrix:');
disp(features);

labels = [1; 0; 1; 0; 1];   % Example labels

model = fitcsvm(features, labels);
predictions = predict(model, features);

figure;
cm = confusionchart(labels, predictions);
cm.Title = 'ECG Classification Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';