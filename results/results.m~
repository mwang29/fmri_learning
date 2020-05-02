%% Only test vs. test and retest (no subject constraint)
all_data = csvread('HCP100_E200_LR0.001_R0_S0_Y1.csv', 1, 0);
test_only = csvread('HCP100_E200_LR0.001_R0_S0_Y1_test.csv', 1, 0);
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
%% 5 replicates per pct comp, only on test subjects

rep1 = csvread('HCP100_E200_LR0.001_R0_S0_Y1_1_test.csv',1,0);
rep2 = csvread('HCP100_E200_LR0.001_R0_S0_Y1_2_test.csv',1,0);
rep3 = csvread('HCP100_E200_LR0.001_R0_S0_Y1_3_test.csv',1,0);
rep4 = csvread('HCP100_E200_LR0.001_R0_S0_Y1_4_test.csv',1,0);
rep5 = csvread('HCP100_E200_LR0.001_R0_S0_Y1_5_test.csv',1,0);

pct_comps = rep1(:,1);
accuracy = [rep2(:,2), rep3(:,2), rep4(:,2), rep5(:,2)];
mean(accuracy(1,:))
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
%% Subject Grouping vs. Random Indices
Srep1 = csvread('HCP100_E200_LR0.001_R0_S1_Y1_1.csv',1,0);
Srep2 = csvread('HCP100_E200_LR0.001_R0_S1_Y1_2.csv',1,0);
Rrep1 = csvread('HCP100_E200_LR0.001_R0_S0_Y1.csv', 1, 0);
Rrep2 = csvread('HCP100_E200_LR0.001_R0_S0_Y1_1.csv', 1, 0);

pct_comps = Srep1(:,1);
subj_acc = [Srep1(:,2), Srep2(:,2)];
random_acc = [Rrep1(:,2), Rrep2(:,2)];
subj_loss = [Srep1(:,3), Srep2(:,3)];
random_loss = [Rrep1(:,3), Rrep2(:,3)];

avg_subj_acc = mean(mean(subj_acc));
avg_random_acc = mean(mean(random_acc));
std_subj_acc = mean(std(subj_acc));
std_random_acc = mean(std(random_acc));

fig1 = subplot(1,2,1);
set(gcf, 'Position',  [100, 100, 1250, 500])
p1 = plot(pct_comps, random_loss, 'r.', 'MarkerSize',15); hold on;
p2 = plot(pct_comps, subj_loss, 'b.', 'MarkerSize',15);
title('Proportion PCs vs. Validation Loss, Subject Grouping')
xlabel('Proportion PCs'), ylabel('Validation Loss')
h = [p1(1);p2];
legend(h, 'Random Indices', 'Subject Grouping') 
axis square

subplot(1,2,2);
p3 = plot(pct_comps, random_acc, 'r.', 'MarkerSize',15); hold on;
p4 = plot(pct_comps, subj_acc, 'b.', 'MarkerSize',15);
title('Proportion PCs vs. Accuracy, Subject Grouping')
xlabel('Proportion PCs'), ylabel('Accuracy')
h = [p3(1);p4];
legend(h, 'Random Indices', 'Subject Grouping') 
axis square
saveas(gcf,'pca_subject_grouping.png')

% Random indices are better with lower mean and stdev

%% Tangent space vs. Non tangent space 