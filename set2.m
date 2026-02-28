records = {'100','101','102','103','104'};
features = [];

for i = 1:length(records)
    
    [signal, Fs, tm] = rdsamp(records{i});
    
    ecg = signal(:,1);
    
    % Bandpass filter
    bpFilt = designfilt('bandpassiir', ...
        'FilterOrder', 4, ...
        'HalfPowerFrequency1', 0.5, ...
        'HalfPowerFrequency2', 40, ...
        'SampleRate', Fs);
    
    ecg = filtfilt(bpFilt, ecg);
    
    % R peaks
    [pks, locs] = findpeaks(ecg,'MinPeakDistance',0.6*Fs);
    
    RR = diff(locs)/Fs;
    RR = RR(RR > 0.3 & RR < 2);
    
    % Time domain features
    mean_RR = mean(RR);
    std_RR = std(RR);
    rmssd = sqrt(mean(diff(RR).^2));
    
    features = [features; mean_RR std_RR rmssd];
    
end

labels = [0; 0; 1; 1; 0];   % Example (Normal=0, Arrhythmia=1)

model = fitcsvm(features, labels);

predictions = predict(model, features);

accuracy = sum(predictions == labels)/length(labels) * 100;

fprintf('Model Accuracy: %.2f%%\n', accuracy);

confusionchart(labels, predictions)

figure;
cm = confusionchart(labels, predictions);
cm.Title = 'ECG Arrhythmia Classification';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';