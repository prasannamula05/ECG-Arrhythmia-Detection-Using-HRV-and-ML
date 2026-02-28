ecg = filtered_ecg(:,1);   % Take first channel only

[pks, locs] = findpeaks(ecg, ...
    'MinPeakHeight', 0.5, ...
    'MinPeakDistance', 0.6*fs);

figure
plot(tm(1:2000), ecg(1:2000))
hold on
plot(tm(locs), pks, 'ro')
title('R Peak Detection')