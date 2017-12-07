% This scripts compares the user feedbacks in the two system
clear all
close all

% load user feedbacks in biased and baseline system
data_addr = 'Data-Exp1\';
load([data_addr,'User_study_results'])

%% Exp1 - Biased system
num_kws = size(Selected_keywords,1);
FB_source_biased = Feedbacks_sys_biased;

I_dont_know_biased = FB_source_biased == -1;
mean_biased = zeros(num_kws,1);
var_biased  = zeros(num_kws,1);
std_biased = zeros(num_kws,1);
for kw = 1:num_kws
    indx = ~I_dont_know_biased(kw,:); 
    mean_biased(kw) = mean(FB_source_biased(kw,indx));
    var_biased(kw) = var(FB_source_biased(kw,indx));
    std_biased(kw) = std(FB_source_biased(kw,indx));
end

num_users_biased = size(FB_source_biased,2);
user_var_biased = zeros(num_users_biased,1);
correlation_biased = zeros(1,num_users_biased);
for user = 1:num_users_biased
    indx = ~I_dont_know_biased(:,user);
    correlation_biased(user) = corr(FB_source_biased(indx,user),Machine_estimates(indx));
    user_var_biased(user) = var(FB_source_biased(indx,user));
end
disp(['average correlation to machine estimate in biased system: ', num2str(mean(correlation_biased))])

% number of I don't know answers per users
disp('number of I dont know answers for each user:')
disp(['num of I dont knows in biased system: ',num2str(sum(I_dont_know_biased))])

%% Exp2 - Baseline system
FB_source_baseline = Feedbacks_sys_baseline;
I_dont_know_baseline = FB_source_baseline == -1;
mean_baseline = zeros(num_kws,1);
var_baseline  = zeros(num_kws,1);
std_baseline = zeros(num_kws,1);
for kw = 1:num_kws
    indx = ~I_dont_know_baseline(kw,:);
    mean_baseline(kw) = mean(FB_source_baseline(kw,indx));
    var_baseline(kw) = var(FB_source_baseline(kw,indx));
    std_baseline(kw) = std(FB_source_baseline(kw,indx));
end

num_users_baseline = size(FB_source_baseline,2);
user_var_baseline = zeros(num_users_baseline,1);
correlation_baseline = zeros(1,num_users_baseline);
for user = 1:num_users_baseline
    indx = ~I_dont_know_baseline(:,user);
    correlation_baseline(user) = corr(FB_source_baseline(indx,user),Machine_estimates(indx));
    user_var_baseline(user) = var(FB_source_baseline(indx,user));
end
disp(['average correlation to machine estimate in baseline system: ', num2str(mean(correlation_baseline))])
% number of I don't know answers per users
disp(['num of I dont knows in baseline system: ',num2str(sum(I_dont_know_baseline))])

%% check the significance of the difference
Hypo_kws = zeros(num_kws,1);
P_val_kws = zeros(num_kws,1);
for kw = 1:num_kws
    indx = ~I_dont_know_biased(kw,:); 
    indx2 = ~I_dont_know_baseline(kw,:);
    
    % Test the null hypothesis that the two data vectors are from populations with equal means,
    %  without assuming that the populations also have equal variances.
    [Hypo_kws(kw),P_val_kws(kw)] = ttest2(FB_source_biased(kw,indx),FB_source_baseline(kw,indx2),'Vartype','unequal');
    
    % Two-sided Wilcoxon rank sum test. ranksum tests the null hypothesis that data in x and y are samples 
    % from continuous distributions with equal medians, against the alternative that they are not. 
    % The test assumes that the two samples are independent. x and y can have different lengths.
%     [P_val_kws(kw),Hypo_kws(kw)] = ranksum(FB_source_biased(kw,indx),FB_source_baseline(kw,indx2));
end

%What are the keywords with highest difference in answers in the two systems
[vals,sorted_idx] = sort(P_val_kws);

num_to_show = 30;
indices = sorted_idx(1:num_to_show);
% indices = sorted_idx(num_kws-num_to_show:end);
% indices = Hypo_kws==1;

T = table(Machine_estimates(indices), round(mean_biased(indices)*100)/100,round(mean_baseline(indices)*100)/100,...
    P_val_kws(indices), 'RowNames',Selected_keywords(indices),...
    'VariableNames',{'Machine';'Biased_ave';'Baseline_ave';'P_Value'});
disp(T);

%% Exp1.5 - Biased system AFTER CORRECTION
% First you need to run the main script (select_bias_experiment = true;) to generate these data
load('FB_biased_inferred'); 
FB_source_inferred = FB_biased_inferred;
mean_biased_inferred = zeros(num_kws,1);
var_biased_inferred  = zeros(num_kws,1);
std_biased_inferred = zeros(num_kws,1);
for kw = 1:num_kws
    indx = ~I_dont_know_biased(kw,:); 
    mean_biased_inferred(kw) = mean(FB_source_inferred(kw,indx));
    var_biased_inferred(kw) = var(FB_source_inferred(kw,indx));
    std_biased_inferred(kw) = std(FB_source_inferred(kw,indx));
end

user_var_biased_inferred = zeros(num_users_biased,1);
correlation_biased_inferred = zeros(1,num_users_biased);
for user = 1:num_users_biased
    indx = ~I_dont_know_biased(:,user);
    correlation_biased_inferred(user) = corr(FB_source_inferred(indx,user),Machine_estimates(indx));
    user_var_biased_inferred(user) = var(FB_source_inferred(indx,user));
end
disp(['average correlation to machine estimate in biased system after correction: ', num2str(mean(correlation_biased_inferred))])

%% Plot a figure with x-axis as the p-values and y-axis as the average relevance 
figure;
hold on
plot(P_val_kws(sorted_idx),mean_biased(sorted_idx),'rs')
plot(P_val_kws(sorted_idx),mean_baseline(sorted_idx),'bs')
plot(P_val_kws(sorted_idx),Machine_estimates(sorted_idx),'gs')
% plot(P_val_kws(sorted_idx),mean_biased_inferred(sorted_idx),'ko')
% plot(P_val_kws(sorted_idx),fu_inf_all_alphas(sorted_idx),'r*')
legend( 'biased','baseline','Machine estimate')
for kw =1:num_kws
    plot([P_val_kws(kw),P_val_kws(kw)],[mean_baseline(kw),mean_biased(kw)],'r--');
%     plot([P_val_kws(kw),P_val_kws(kw)],[Machine_estimates(kw),mean_biased(kw)],'g--');
    text(P_val_kws(kw),mean_biased(kw),Selected_keywords(kw),'HorizontalAlignment','right')
end
xlabel('p-value')
ylabel('average relevance')
title('difference between user feedbacks in baseline and biased system')

%% Plotting and analysis
%Average user behaviors in two systems
disp(['num of users: Biased = ', num2str(num_users_biased),', Baseline = ', num2str(num_users_baseline)])

%sort the difference in variance from smallest to largest
diff_var = abs(var_biased - var_baseline);
[~,sorted_var_idx] = sort(diff_var,'descend');

figure;
hold on
plot(diff_var(sorted_var_idx),var_biased(sorted_var_idx),'rs')
plot(diff_var(sorted_var_idx),var_baseline(sorted_var_idx),'bs')
legend( 'biased','baseline')
for kw =1:num_kws
    plot([diff_var(kw),diff_var(kw)],[var_baseline(kw),var_biased(kw)],'r--');
    text(diff_var(kw),var_biased(kw),Selected_keywords(kw),'HorizontalAlignment','right')
end
title('difference between variance of baseline and biased')
xlabel('absolute distance between variances')
ylabel('variance')

disp(['mean variance for baseline: ',num2str(mean(var_baseline)),' and biased: ',num2str(mean(var_biased))])

%% Correlation between machine's estimates and individual users
figure
hold on
h1 = histogram(correlation_baseline);
% h1.Normalization = 'probability';
h1.BinWidth = 0.04;
h1 = histogram(correlation_biased);
% h1.Normalization = 'probability';
h1.BinWidth = 0.04;
h1 = histogram(correlation_biased_inferred);
% h1.Normalization = 'probability';
h1.BinWidth = 0.04;
legend('baseline', 'biased','biased after correction')
plot([mean(correlation_baseline),mean(correlation_baseline)],[0,5],'b--')
plot([mean(correlation_biased),mean(correlation_biased)],[0,5],'r--')
plot([mean(correlation_biased_inferred),mean(correlation_biased_inferred)],[0,5],'y--')
title('correlation to machine estimate')
xlabel('pearson correlation')
disp(['correlation to machine estimate for baseline, biased, and inferred:',...
    num2str(mean(correlation_baseline)),num2str(mean(correlation_biased)),num2str(mean(correlation_biased_inferred))])

%% Create a correlation matrix between all users (of both systems) and plot it as a hitmap
num_total_users = num_users_baseline+num_users_biased;
FB_source_all = [FB_source_baseline,FB_source_biased];
I_dont_know_all = [I_dont_know_baseline,I_dont_know_biased];
corr_matrix = zeros(num_total_users,num_total_users);
for i = 1:num_total_users
    for j = i:num_total_users
        indx_i = ~I_dont_know_all(:,i);
        indx_j = ~I_dont_know_all(:,j);
        indx = indx_i & indx_j;
        corr_matrix(i,j) = corr(FB_source_all(indx,i),FB_source_all(indx,j));
        corr_matrix(j,i) = corr_matrix(i,j);
    end
end
figure
imagesc(corr_matrix, [0,1]); 
colormap Gray
colorbar
hold on
plot([num_users_baseline+0.5,num_users_baseline+1],[0,num_total_users+0.5],'r')
plot([0,num_total_users+0.5],[num_users_baseline+0.5,num_users_baseline+0.5],'r')
title('Correlation between all users')
