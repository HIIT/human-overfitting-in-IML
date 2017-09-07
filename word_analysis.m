clear all
close all


% load user feedbacks in biased and baseline system
data_addr = 'Data-Exp1\';
load([data_addr,'User_study_results'])

%% Exp1 - Biased system
% people_who_said_NO = [2,3,6,11,14,15,18,19,20,21];

num_kws = size(Selected_keywords,1);

FB_source_biased = Feedbacks_sys_biased;
% The following participants are filtered because they did not answer
% 2/3 of questions or they time was less than 3 mins
filtered_users_biased = [12,13,18]; %12,13,17,18
FB_source_biased(:,filtered_users_biased) = [];

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
disp(['average correlation in biased system: ', num2str(mean(correlation_biased))])

% number of I don't know answers per users
disp('number of I dont know answers for each user:')
disp(['num of I dont knows in biased system: ',num2str(sum(I_dont_know_biased))])

%% Exp1.5 - Biased system AFTER CORRECTION

load('FB_biased_inferred');
FB_source_inferred = FB_biased_inferred;
FB_source_inferred(:,filtered_users_biased) = [];
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
disp(['average correlation in biased system after correction: ', num2str(mean(correlation_biased_inferred))])


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
disp(['average correlation in baseline system: ', num2str(mean(correlation_baseline))])
% number of I don't know answers per users
disp(['num of I dont knows in baseline system: ',num2str(sum(I_dont_know_baseline))])

%% Plotting and analysis
%Average user behaviors in two systems
figure
hold on
plot([mean_baseline,mean_biased,mean_biased_inferred,Machine_estimates],'s')
legend('baseline', 'biased','biased_inferred','Machine')
for kw =1:num_kws
    plot([kw,kw],[mean_baseline(kw),mean_biased(kw)],'r');
end
xlabel('keywords')
ylabel('average users feedback (probability of relevance)')

%% check the significance of the difference
Hypo_kws = zeros(num_kws,1);
P_val_kws = zeros(num_kws,1);
for kw = 1:num_kws
    indx = ~I_dont_know_biased(kw,:); 
    indx2 = ~I_dont_know_baseline(kw,:);
    % Test the null hypothesis that the two data vectors are from populations with equal means,
    %  without assuming that the populations also have equal variances.
    [Hypo_kws(kw),P_val_kws(kw)] = ttest2(FB_source_biased(kw,indx),FB_source_baseline(kw,indx2),'Vartype','unequal');
end

%What are the keywords with highest difference in answers in the two systems
[vals,sorted_idx] = sort(P_val_kws);

num_to_show = 30;
indices = sorted_idx(1:num_to_show);
% indices = sorted_idx(num_kws-num_to_show:end);
% indices = Hypo_kws==1;
% T = table(Machine_estimates(indices), mean_biased(indices),mean_baseline(indices),...
%     Hypo_kws(indices), P_val_kws(indices), 'RowNames',Selected_keywords(indices),...
%     'VariableNames',{'Machine';'Biased_ave';'Baseline_ave';'Sig_Diff';'P_Value'});
T = table(Machine_estimates(indices), mean_biased(indices),mean_baseline(indices),mean_biased_inferred(indices),...
    P_val_kws(indices), 'RowNames',Selected_keywords(indices),...
    'VariableNames',{'Machine';'Biased_ave';'Baseline_ave';'Inferred_ave';'P_Value'});
disp(T);

% Plot a figure with x-axis as the p-values and y-axis as the average relevance 
figure;
hold on
plot(P_val_kws(sorted_idx),mean_biased(sorted_idx),'rs')
plot(P_val_kws(sorted_idx),mean_baseline(sorted_idx),'bs')
plot(P_val_kws(sorted_idx),Machine_estimates(sorted_idx),'gs')
legend( 'biased','baseline','Machine estimate')
for kw =1:num_kws
    plot([P_val_kws(kw),P_val_kws(kw)],[mean_baseline(kw),mean_biased(kw)],'r--');
%     plot([P_val_kws(kw),P_val_kws(kw)],[Machine_estimates(kw),mean_biased(kw)],'g--');
    text(P_val_kws(kw),mean_biased(kw),Selected_keywords(kw),'HorizontalAlignment','right')
end
xlabel('p-value')
ylabel('average relevance')
title('difference between user feedbacks in baseline and biased system')

% This is a good figure (at least better than the last one)

% %Plot the variances
% figure
% hold on
% h3 = histogram(var_baseline);
% % h1.Normalization = 'probability';
% h3.BinWidth = 0.01;
% h3 = histogram(var_biased);
% % h1.Normalization = 'probability';
% h3.BinWidth = 0.01;
% legend('var baseline', 'var biased')
% plot([mean(var_baseline),mean(var_baseline)],[0,5],'b--')
% plot([mean(var_biased),mean(var_biased)],[0,5],'r--')
% title('histogram of variances')
% xlabel('variance')

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
%     plot([P_val_kws(kw),P_val_kws(kw)],[Machine_estimates(kw),mean_biased(kw)],'g--');
    text(diff_var(kw),var_biased(kw),Selected_keywords(kw),'HorizontalAlignment','right')
end
title('different between variance of baseline and biased')
xlabel('distance')
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
%% average correlation between mean reasults to machine's estimate
Rho_between_means = zeros(6,1);

Rho_between_means(1) = corr(mean_biased,mean_baseline);
Rho_between_means(2) = corr(mean_baseline,Machine_estimates);
Rho_between_means(3) = corr(mean_biased,Machine_estimates);
Rho_between_means(4) = corr(mean_biased_inferred,mean_baseline);
Rho_between_means(5) = corr(mean_biased_inferred,mean_biased);
Rho_between_means(6) = corr(mean_biased_inferred,Machine_estimates);
disp('Rho between mean of all users: biased and baseline, baseline and machine, biased and machine, inferred and baseline, inferred and biased, inferred and machine')
disp(Rho_between_means')


%% For each user, compute average corr between and within
% first compute within group correlation for each user
corr_within_baseline = zeros(num_users_baseline,num_users_baseline);
for i = 1:num_users_baseline
    for j = 1:num_users_baseline
        indx_i = ~I_dont_know_baseline(:,i);
        indx_j = ~I_dont_know_baseline(:,j);
        indx = indx_i & indx_j;
        corr_within_baseline(i,j) = corr(FB_source_baseline(indx,i),FB_source_baseline(indx,j));
    end
end
figure
subplot(3,1,1) 
title('Within user group correlation for baseline system')
imagesc(corr_within_baseline, [0,1]);     

corr_within_biased = zeros(num_users_biased,num_users_biased);
for i = 1:num_users_biased
    for j = 1:num_users_biased
        indx_i = ~I_dont_know_biased(:,i);
        indx_j = ~I_dont_know_biased(:,j);
        indx = indx_i & indx_j;
        corr_within_biased(i,j) = corr(FB_source_biased(indx,i),FB_source_biased(indx,j));
    end
end
subplot(3,1,2) 
title('Within user group correlation for biased system')
imagesc(corr_within_biased, [0,1]);             

corr_within_infered = zeros(num_users_biased,num_users_biased);
for i = 1:num_users_biased
    for j = 1:num_users_biased
        indx_i = ~I_dont_know_biased(:,i);
        indx_j = ~I_dont_know_biased(:,j);
        indx = indx_i & indx_j;
        corr_within_infered(i,j) = corr(FB_source_inferred(indx,i),FB_source_inferred(indx,j));
    end
end
subplot(3,1,3) 
title('Within user group correlation for biased system')
imagesc(corr_within_infered, [0,1]);     


% second, calculate between users correlations
% biased and baseline:
corr_between_bas_bias = zeros(num_users_baseline,num_users_biased);
for i = 1:num_users_baseline
    for j = 1:num_users_biased
        indx_i = ~I_dont_know_baseline(:,i);
        indx_j = ~I_dont_know_biased(:,j);
        indx = indx_i & indx_j;
        corr_between_bas_bias(i,j) = corr(FB_source_baseline(indx,i),FB_source_biased(indx,j));
    end
end
figure
subplot(3,1,1) 
title('between user group correlation of baseline and biased system')
imagesc(corr_between_bas_bias, [0,1]);  

% inferred and baseline:
corr_between_bas_inf = zeros(num_users_baseline,num_users_biased);
for i = 1:num_users_baseline
    for j = 1:num_users_biased
        indx_i = ~I_dont_know_baseline(:,i);
        indx_j = ~I_dont_know_biased(:,j);
        indx = indx_i & indx_j;
        corr_between_bas_inf(i,j) = corr(FB_source_baseline(indx,i),FB_source_inferred(indx,j));
    end
end
subplot(3,1,2) 
title('between user group correlation of baseline and inferred system')
imagesc(corr_between_bas_inf, [0,1]); 

% inferred and biased:
corr_between_bias_inf = zeros(num_users_biased,num_users_biased);
for i = 1:num_users_biased
    for j = 1:num_users_biased
        indx_i = ~I_dont_know_biased(:,i);
        indx_j = ~I_dont_know_biased(:,j);
        indx = indx_i & indx_j;
        corr_between_bias_inf(i,j) = corr(FB_source_biased(indx,i),FB_source_inferred(indx,j));
    end
end
subplot(3,1,3) 
title('between user group correlation of biased and inferred system')
imagesc(corr_between_bias_inf, [0,1]); 


%% Create a between within corr figure!

% BIASED VS BASELINE
ave_between_corr_bias = mean(corr_between_bas_bias,1);
ave_between_corr_bas = mean(corr_between_bas_bias,2);

ave_within_cor_bias = (sum(corr_within_biased)-1)./(num_users_biased-1);
ave_within_cor_bas = (sum(corr_within_baseline)-1)./(num_users_baseline-1);
ave_within_cor_inf = (sum(corr_within_infered)-1)./(num_users_biased-1);

figure
xlabel('within group correlation')
ylabel('between groups correlation')
hold on
plot([ave_within_cor_bas],[ave_between_corr_bas'],'sr')
plot([ave_within_cor_bias],[ave_between_corr_bias],'sb')
legend('Baseline system','Biased system')

% INFERRED VS BASELINE
ave_between_corr_inf = mean(corr_between_bas_inf,1);
ave_between_corr_bas2 = mean(corr_between_bas_inf,2);

figure
xlabel('within group correlation')
ylabel('between groups correlation')
hold on
plot([ave_within_cor_bas],[ave_between_corr_bas2'],'sr')
plot([ave_within_cor_inf],[ave_between_corr_inf],'sb')
legend('Baseline system','Inferred system')




%% Correlation between Baseline and biased, and Baseline and inferred
figure
hold on
h2 = histogram(ave_between_corr_inf);
h2.BinWidth = 0.02;
h2 = histogram(ave_between_corr_bias);
h2.BinWidth = 0.02;

legend('Inferred', 'Biased')
plot([mean(ave_between_corr_inf),mean(ave_between_corr_inf)],[0,3],'b--')
plot([mean(ave_between_corr_bias),mean(ave_between_corr_bias)],[0,3],'r--')
title('correlation to baseline users')
xlabel('pearson correlation')


%%% CAN WE DO THESE CORRELATIN ANALYSIS IN THE WORD LEVEL RATHER THAN FULL
%%% SET OF WORDS CORRELATION?