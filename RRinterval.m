% RR intervals
RR = diff(locs) / fs;

% Time-domain features
mean_RR = mean(RR);
std_RR = std(RR);
rmssd = sqrt(mean(diff(RR).^2));

% Heart Rate
HR = 60 ./ RR;
mean_HR = mean(HR);

fprintf('Mean RR: %.4f\n', mean_RR);
fprintf('STD RR: %.4f\n', std_RR);
fprintf('RMSSD: %.4f\n', rmssd);
fprintf('Mean HR: %.2f bpm\n', mean_HR);