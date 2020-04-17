loss200 = csvread('loss_200.csv', 1, 0);
loss100 = csvread('loss.csv', 1, 0);
acc200 = csvread('accuracies_200.csv', 1, 0);
acc100 = csvread('accuracies.csv', 1, 0);


model_loss = fitlm(loss200(:,1),loss200(:,2));
model_acc = fitlm(acc200(:,1),acc200(:,2));

fig = subplot(1,2,1);
plot(loss200(:,1), loss200(:,2), '.', 'MarkerSize',15);
title('Proportion PCs vs. Validation Loss')
xlabel('Proportion PCs'), ylabel('Validation Loss')

subplot(1,2,2);
plot(acc200(:,1), acc200(:,2), '.', 'MarkerSize',15);
title('Proportion PCs vs. Test Accuracy')
xlabel('Proportion PCs'), ylabel('Test Accuracy')

