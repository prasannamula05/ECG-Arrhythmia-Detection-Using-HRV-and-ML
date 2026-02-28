RR = diff(locs) / fs;

% Remove abnormal intervals
RR = RR(RR > 0.3 & RR < 2);   % Keep physiologically valid RR (30–200 bpm)

RR = RR(~isnan(RR));
RR = RR(~isinf(RR));

meanRR = mean(RR);

fprintf("Mean RR: %.4f\n", meanRR);

Fs_rr = 1/meanRR;

[pxx,f] = pwelch(RR - mean(RR),[],[],[],Fs_rr);

figure
plot(f,10*log10(pxx))
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
title('HRV Power Spectrum')