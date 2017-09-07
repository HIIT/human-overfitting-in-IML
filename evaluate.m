clear all
close all


% Load the user experiment data
load('user_study_mse_results')

num_methods = size(Method_list,2);
num_iterations = size(Loss_1,2);
num_users = size(Loss_1,3);

%display useful information about the simulation
disp(struct2table(sparse_params));
disp('Feedback is on the probability of relevance of features');

disp(['Number of features: ', num2str(num_features),'.']);
disp(['Number of training data: ', num2str(num_trainingdata),'.']);
disp(['Number of users ', num2str(num_users), ' runs.']);

figure
Loss_1_mean = mean(Loss_1,3)';
%This one is the data
% plot([0:num_iterations-1],Loss_1_mean,'.-','LineWidth',2);
plot([1:num_iterations],Loss_1_mean,'.-','LineWidth',2);
legend(Method_list)
title('Loss function')
xlabel('Number of Expert Feedbacks')
ylabel('Mean Squared Error')

%statistical analysis of improvements
GT_methods_loss_1 = Loss_1(:,1,:);
%TODO: find methods that have zero std and call them gt methods first
GT_methods_loss_1 = reshape(GT_methods_loss_1,num_methods,num_users);
figure
hold on
for m=1:num_methods
    errors = GT_methods_loss_1(m,:);
    h1 = histogram(errors);
    h1.Normalization = 'probability';
    h1.BinWidth = 0.05;
end
disp('Mean values: No FB, orig FB, inf FB')
disp(mean(GT_methods_loss_1'));
disp('STD values: No FB, orig FB, inf FB')
disp(std(GT_methods_loss_1'));
disp(Method_list)

title('Histogram of Loss function')
xlabel('MSE')
legend(Method_list)

%Fin the targetter methods:
method_OF_user = find(strcmp('User FB before correction', Method_list));
method_inferred_user = find(strcmp('User FB after correction', Method_list));
disp('is the inferred results better?')
% disp( GT_methods_loss_1(method_OF_user,:) - GT_methods_loss_1(method_inferred_user,:))
%I guess h should be one
[h,p,ci,stats] = ttest(GT_methods_loss_1(method_OF_user,:),GT_methods_loss_1(method_inferred_user,:)) %, 'Alpha',0.05

figure
hold on
h2 = histogram(GT_methods_loss_1(method_OF_user,:) - GT_methods_loss_1(method_inferred_user,:));
h2.BinWidth = 0.005;
title(['Correction effect for ' num2str(num_users) ' users'])
xlabel('Decreaase in MSE after bias correction')
plot([0,0],[0,2],'r--')
% figure
% %for real data case Loss_2 is MSE on training in the normalized space
% Loss_2_mean = mean(Loss_2,3)';
% %This one is the data
% plot([1:num_iterations], Loss_2_mean,'.-','LineWidth',2);
% legend(Method_list)
% title('Loss function')
% xlabel('Number of Expert Feedbacks')
% ylabel('Mean Squared Error on Training')
