%this script evaluates the MSE results of the experiments (after running main.m)

clear all
close all

% Load the user experiment data
load('user_study_mse_results')

num_methods = size(Method_list,2);
num_iterations = size(Loss_1,2); %iterations are not used in this study
num_users = size(Loss_1,3);

%display useful information about the simulation
disp(struct2table(sparse_params));
disp('Feedback is on the probability of relevance of features');

disp(['Number of features: ', num2str(num_features),'.']);
disp(['Number of training data: ', num2str(num_trainingdata),'.']);
disp(['Number of users ', num2str(num_users), '.']);

figure
Loss_1_mean = mean(Loss_1,3)';
plot([1:num_iterations],Loss_1_mean,'.-','LineWidth',2);
legend(Method_list)
title('Loss function')
xlabel('Number of Expert Feedbacks')
ylabel('Mean Squared Error')

%statistical analysis of improvements
methods_loss_1 = Loss_1(:,1,:);  %iterations are not used in this study
methods_loss_1 = reshape(methods_loss_1,num_methods,num_users);
figure
hold on
for m=1:num_methods
    errors = methods_loss_1(m,:);
    h1 = histogram(errors);
    h1.Normalization = 'probability';
    h1.BinWidth = 0.05;
end
disp('Mean values: No FB, orig FB, inf FB')
disp(mean(methods_loss_1'));
disp('STD values: No FB, orig FB, inf FB')
disp(std(methods_loss_1'));
disp(Method_list)

title('Histogram of Loss function')
xlabel('MSE')
legend(Method_list)

%Find the target methods:
method_OF_user = find(strcmp('User FB before correction', Method_list));
method_inferred_user = find(strcmp('User FB after correction', Method_list));
method_no_feedback = find(strcmp('no feedback', Method_list)); 

%Test 1: feedback is better than no feedback
disp('Is receiving feedback in general better than not receiving it?')
[h,p,ci,stats] = ttest(methods_loss_1(method_no_feedback,:),methods_loss_1(method_OF_user,:)); %, 'Alpha',0.05
if h
    disp(['Yes, with p-value=',num2str(p), ' and CI=',num2str(ci) ]);
else
    disp(['No, with p-value=',num2str(p), ' and CI=',num2str(ci) ]);
end

%Test 2: user model is better than directly using the feedbacks
disp('Is the inferred results better than feedback?')
[h,p,ci,stats] = ttest(methods_loss_1(method_OF_user,:),methods_loss_1(method_inferred_user,:)); %, 'Alpha',0.05
if h
    disp(['Yes, with p-value=',num2str(p), ' and CI=',num2str(ci) ]);
else
    disp(['No, with p-value=',num2str(p), ' and CI=',num2str(ci) ]);
end

figure
hold on
diffs = methods_loss_1(method_OF_user,:) - methods_loss_1(method_inferred_user,:);
h2 = histogram(diffs);
h2.BinWidth = 0.006;
title(['Correction effect for ' num2str(num_users) ' users'])
xlabel('Mean Squared Error change','FontSize',16)
ylabel('Number of Participants','FontSize',16)
plot([0,0],[0,4],'r--')

%training data error
figure
%Loss_2 is MSE on training in the normalized space
Loss_2_mean = mean(Loss_2,3)';
plot([1:num_iterations], Loss_2_mean,'.-','LineWidth',2);
legend(Method_list)
title('Loss function')
xlabel('Number of Expert Feedbacks')
ylabel('Mean Squared Error on Training')


%% The following code groups the results based on the participants asnwer
% to the question "Did you find the machine's estimates useful?"
% Please note that this question was only asked in the biased experiment

data_addr = 'Data-Exp1\';
% load user feedbacks in biased and baseline system
load([data_addr,'User_study_results'])

% dumb way to check if the MSE data are for the biased system:
if size(Feedbacks_sys_biased_did_u_use,2) == num_users
    %of course if the number of participants is the same, the condition is
    %always accepted!
    
%     disp('Mean values: No FB, orig FB, inf FB FOR YES')
%     disp(mean(methods_loss_1(:,Feedbacks_sys_biased_did_u_use==1)'));
%     disp('STD values: No FB, orig FB, inf FB FOR YES')
%     disp(std(methods_loss_1(:,Feedbacks_sys_biased_did_u_use==1)'));
% 
%     disp('Mean values: No FB, orig FB, inf FB FOR NO')
%     disp(mean(methods_loss_1(:,Feedbacks_sys_biased_did_u_use==0)'));
%     disp('STD values: No FB, orig FB, inf FB FOR NO')
%     disp(std(methods_loss_1(:,Feedbacks_sys_biased_did_u_use==0)'));
    
    imp_users_yes = diffs(Feedbacks_sys_biased_did_u_use==1); 
    imp_users_no = diffs(Feedbacks_sys_biased_did_u_use==0); 

    %plot the histogram of two groups
    figure
    hold on
    h1 = histogram(imp_users_no);
    h1.BinWidth = 0.006;
    h2 = histogram(imp_users_yes);
    h2.BinWidth = 0.006;
    title(['Correction effect for ' num2str(num_users) ' users'])
    xlabel('Mean Squared Error change','FontSize',16)
    ylabel('Number of Participants','FontSize',16)
    plot([0,0],[0,4],'r--')

    %plot the stacked histogram (using stacked bars) 
    figure   
    binrng = 0:0.006:0.09;
    counts1 = histcounts(imp_users_no, binrng);                                 
    counts2 = histcounts(imp_users_yes, binrng);                                                                
    new_bings = binrng+0.003;
    new_bings = new_bings(1:end-1); 
    bar(new_bings,[counts1;counts2]','stacked','BarWidth',1)
    hold on
    plot([0,0],[0,4],'r--')
    hold off
    legend('Machine estimates were not really useful, I did not consider them much',...
        'Machine estimates were useful, I have considered them when giving some of the answers')
    xlabel('Mean Squared Error change','FontSize',16)
    ylabel('Number of Participants','FontSize',16)  
    % legend boxoff
end
