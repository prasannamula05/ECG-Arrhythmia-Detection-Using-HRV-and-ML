fs = Fs;   % sampling frequency

% Design bandpass filter
bpFilt = designfilt('bandpassiir', ...
    'FilterOrder', 4, ...
    'HalfPowerFrequency1', 0.5, ...
    'HalfPowerFrequency2', 40, ...
    'SampleRate', fs);

filtered_ecg = filtfilt(bpFilt, signal);

% Plot comparison
figure
subplot(2,1,1)
plot(tm(1:2000), signal(1:2000))
title('Raw ECG')

subplot(2,1,2)
plot(tm(1:2000), filtered_ecg(1:2000))
title('Filtered ECG')

