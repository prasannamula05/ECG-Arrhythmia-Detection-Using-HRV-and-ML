[ann, anntype] = rdann('100','atr');
unique(anntype)

window = round(0.3 * Fs);   % 300 ms window

beats = [];

for k = 1:length(ann)
    
    idx = ann(k);
    
    if idx-window > 0 && idx+window < length(ecg)
        beat = ecg(idx-window : idx+window);
        beats = [beats; beat'];
    end
    
end

figure;
for i = 1:5
    subplot(5,1,i)
    plot(beats(i,:))
    title(['Beat ', num2str(i)])
end

labels = zeros(length(anntype),1);

for k = 1:length(anntype)
    
    if anntype(k) == 'N'
        labels(k) = 0;   % Normal
    else
        labels(k) = 1;   % Abnormal
    end
    
end

normal_idx = find(labels == 0, 1);
abnormal_idx = find(labels == 1, 1);

figure;
subplot(2,1,1)
plot(beats(normal_idx,:))
title('Normal Beat')

subplot(2,1,2)
plot(beats(abnormal_idx,:))
title('Abnormal Beat')

[coeff,score,~,~,explained] = pca(beats);

figure;
plot(cumsum(explained))
xlabel('Number of Components')
ylabel('Variance Explained (%)')

model = fitcsvm(features, labels);

CVmodel = crossval(model, 'KFold', 5);
loss = kfoldLoss(CVmodel);

accuracy = (1 - loss) * 100;
fprintf('Cross-Validated Accuracy: %.2f%%\n', accuracy);