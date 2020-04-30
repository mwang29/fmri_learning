%% Only test vs. test and retest
all_data = csvread('HCP100_E200_LR0.001_R0_S0_Y1.csv', 1, 0);
test_only = csvread('HCP100_E200_LR0.001_R0_S1_Y1.csv', 1, 0);
pct_comps = all_data(:,1);
accuracies = [all_data(:,2), test_only(:,2)];
losses = [all_data(:,3), test_only(:,3)];

fig1 = subplot(1,2,1);
set(gcf, 'Position',  [100, 100, 1250, 500])
plot(pct_comps, losses, '.', 'MarkerSize',15);
title('Proportion PCs vs. Validation Loss')
xlabel('Proportion PCs'), ylabel('Validation Loss')
legend('Test and Retest', 'Test only') 
axis square

subplot(1,2,2);
plot(pct_comps, accuracies, '.', 'MarkerSize',15);
title('Proportion PCs vs. Test Accuracy')
xlabel('Proportion PCs'), ylabel('Test Accuracy')
legend('Test and Retest', 'Test only') 
saveas(gcf,'pca_test_retest.png')
axis square
%% 5 replicates per pct comp

rep1 = csvread('HCP100_E200_LR0.001_R0_S1_Y1_1.csv',1,0);
rep2 = csvread('HCP100_E200_LR0.001_R0_S1_Y1_2.csv',1,0);
rep3 = csvread('HCP100_E200_LR0.001_R0_S1_Y1_3.csv',1,0);
rep4 = csvread('HCP100_E200_LR0.001_R0_S1_Y1_4.csv',1,0);
rep5 = csvread('HCP100_E200_LR0.001_R0_S1_Y1_5.csv',1,0);

pct_comps = rep1(:,1);
accuracy = [rep2(:,2), rep3(:,2), rep4(:,2), rep5(:,2)];
loss = [rep2(:,3), rep3(:,3), rep4(:,3), rep5(:,3)];

fig2 = subplot(1,2,1);
set(gcf, 'Position',  [100, 100, 1250, 500])
plot(pct_comps, loss, 'r.', 'MarkerSize',15);
title('Proportion PCs vs. Validation Loss')
xlabel('Proportion PCs'), ylabel('Validation Loss')
axis square

subplot(1,2,2);
plot(pct_comps, accuracy, 'b.', 'MarkerSize',15);
title('Proportion PCs vs. Test Accuracy')
xlabel('Proportion PCs'), ylabel('Test Accuracy')
axis square
saveas(gcf,'pca_reps.png')
%% 