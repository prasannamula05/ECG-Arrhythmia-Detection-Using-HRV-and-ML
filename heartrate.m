RR_intervals = diff(locs) / fs;
heart_rate = 60 ./ RR_intervals;

mean_HR = mean(heart_rate)