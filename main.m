% This script is designed for the second round of the overfitting experiments
% We assume that the users give feedback about the Pr. of relevance

close all
clear all
rng(54321)
RNG_SEED = rng;

%% Load the proper dataset

data_addr = 'Data-Exp1\';
% load 'X_train', 'Y_train', 'X_test', 'Y_test','y_mean','y_std'
% X and Y for training are in the normalized space
% mean and std of training of y in original space is saved in y_mean and y_std
load([data_addr,'AllData-Exp1']);
% load indices that are excluded from the user study (only 70 kws were used)
load([data_addr,'I_dont_know_indices_amazon']) 
% load user feedbacks in biased and baseline system
load([data_addr,'User_study_results'])

%% Select the experiment that you want to analyze (biased or baseline?):

%set the following to false to run the study for the baseline experiment
select_bias_experiment = true;

if select_bias_experiment
    FB_source = Feedbacks_sys_biased;
else
    FB_source = Feedbacks_sys_baseline;
end

%% Parameters and data setup

%data parameters
num_features       = size(X_train,1);
num_trainingdata   = size(X_train,2);  
num_users          = size(FB_source,2);
Feedback_all = FB_source;

FB_biased_inferred = zeros(size(FB_source,1), num_users); %dummy
% validation_indices is required in the post analyses (alpha CV model)
validation_indices =  1:size(Y_test,1);%datasample(1:size(Y_test,1),1000,'Replace',false); %TODO: change it to the highest amount
figure %this is needed for plotting the CV results of the alpha CV model
num_iterations   = 70; %total number of user feedback (not used for now)

%Features that were used in the user study
considered_kws = setdiff(1:num_features,I_dont_know_indices)';

%model parameters (similar to Mach Lear (2017) paper)
sparse_options = struct('damp', 0.8, ...
              'damp_decay', 0.95, ...
              'robust_updates', 2, ...
              'verbosity', 0, ...
              'max_iter', 1000, ...
              'gamma_threshold', 1e-4, ...
              'w_threshold', 1e-5, ...
              'min_site_prec', 1e-6, ...
              'max_site_prec', Inf, ...
              'w_mean_update_threshold', 1e-6, ...
              'w_prec_update_threshold', 1e6, ...
              'hermite_n', 11, ...
              'degenerate_representation', 0, ...
              'si', []);
sparse_params = struct('kappa_prior', 0, ...
              'p_u', 0.99, ...  % Maybe we can directly ask the users about this
              'rho_prior', 0, ...
              'rho', 0.3, ...
              'sigma2_prior', 0, ...
              'sigma2', 1, ...  %This is from CV results (in final version there should be a prior on this with sigma2_a and _b set to 1
              'tau2_prior', 0, ...
              'tau2', 0.1^2, ...
              'eta2', 0.1^2); % (NOT USED IN case where feedback is on relevance)

% Put distribution assumptions on sigma2 
sparse_params.sigma2_prior  = 1;
sparse_params.sigma2_a  = 1;
sparse_params.sigma2_b  = 1;

%% METHOD LIST
% NOTE: in this study we are not considering Sequentiality of methods.

% Set the desirable methods to 'True' and others to 'False'. 
% only the 'True' methods will be considered in the simulation
METHODS = {
     'True',   'no feedback';
     'True',   'User FB before correction';
     'True',   'User FB after correction';     
     'False',   'User FB after correction - alpha CV model';
     }; 

Method_list = [];
for m = 1:size(METHODS,1)
    if strcmp(METHODS(m,1),'True')
        Method_list = [Method_list,METHODS(m,2)];
    end
end
%number of methods that we want to consider
num_methods = size(Method_list,2); 

%% Main algorithm
Loss_1 = zeros(num_methods, num_iterations, num_users); % MSE on test
Loss_2 = zeros(num_methods, num_iterations, num_users); % MSE on train

%Calculate the posterior before observing any feedbacks
posterior_no_fb = calculate_posterior(X_train, Y_train, [], ...
     sparse_params, sparse_options);
Machine_estimates = posterior_no_fb.p(considered_kws);
Machine_estimates = round(100*Machine_estimates)/100; %this is how we show the values to the users
best_alphas = zeros(num_users,1);
tic

for user = 1:num_users
    disp(['user number ', num2str(user), ' from ', num2str(num_users), '. acc time = ', num2str(toc) ]);

    for method_num = 1:num_methods
        method_name = Method_list(method_num);
%         disp(['method=', method_name]);

        %Feedback = values (1st column) and indices (2nd column) of user feedback
        Feedback = [];            %only used in experimental design methdos        
        sparse_options.si = [];   % carry prior site terms between interactions
        %% Calculate ground truth solutions
            
        if find(strcmp('no feedback', method_name))
            posterior = posterior_no_fb;
        end
        
        
        if find(strcmp('User FB before correction', method_name))
            %load real feedback of the user
            Feedback = [Feedback_all(:,user),considered_kws];
            posterior = calculate_posterior(X_train, Y_train, Feedback, ...
                sparse_params, sparse_options);
        end
        
        
        if find(strcmp('User FB after correction', method_name))
            tr_p = posterior_no_fb.p(considered_kws);
            % user's update knowledge based on marginal inclusion probabilities
            %load real feedback of the user
            fu_upd = Feedback_all(:,user);
            I_dont_knows = fu_upd == -1;
            %Infer user hidden likelihood
            r_tmp = (1 - tr_p) ./ tr_p;
            ba = (1 - fu_upd) ./ fu_upd ./ r_tmp;
            fu_inf = 1 ./ (1 + ba);
            fu_inf(I_dont_knows) = -1;
            FB_biased_inferred(:,user) = fu_inf;
            %If the assumptions about user behavior are correct, then fu_inf is the hidden fu.
            Feedback = [fu_inf,considered_kws];
            posterior = calculate_posterior(X_train, Y_train, Feedback, ...
                sparse_params, sparse_options);
        end
        
        
        if find(strcmp('User FB after correction - alpha CV model', method_name))
            tr_p = posterior_no_fb.p(considered_kws);
            
            %load real feedback of the user
            fu_upd = Feedback_all(:,user);
            I_dont_knows = fu_upd == -1;
            %Infer user hidden likelihood assume that:
            % fu_inf = f (1 - p)^a / ((f (1 - p)^a + (1 - f) p^a)
            % learn alpha through CV on some validation set
            %What is the best alpha for the current user?
            alphas = 0:0.1:2;
            mse_val = zeros(size(alphas,2),1);
            for i = 1:size(alphas,2)
                %infer the feedback for the given alpha
                numerator = fu_upd .* (1-tr_p).^alphas(i);
                denominator = numerator + (1-fu_upd) .* tr_p.^alphas(i);
                fu_inf = numerator./denominator;
                fu_inf(I_dont_knows) = -1;
                %compute the posterior
                posterior_temp = calculate_posterior(X_train, Y_train, [fu_inf,considered_kws], ...
                    sparse_params, sparse_options);
                %calculate validation error on part of the test data
                Y_hat_val = X_test(:,validation_indices)'*posterior_temp.mean;
                Y_hat_val = Y_hat_val .* y_std + y_mean;
                mse_val(i) =  mean((Y_hat_val- Y_test(validation_indices)).^2);
%                %calculate validation error on the training data
%                Y_hat_train = X_train'*posterior_temp.mean;
%                mse_val(i) =  mean((Y_hat_train- Y_train).^2);
            end
            [best_mse_val,idx] = min(mse_val);
            best_alphas(user) = alphas(idx);
            disp(['best alpha for user ',num2str(user),' is ', num2str(alphas(idx)) ]);
            subplot(ceil(num_users/4),4,user)
            plot(alphas,mse_val,'.-');
            numerator = fu_upd .* (1-tr_p).^alphas(idx);
            denominator = numerator + (1-fu_upd) .* tr_p.^alphas(idx);
            fu_inf = numerator./denominator;
            fu_inf(I_dont_knows) = -1;
%            FB_biased_inferred(:,user) = fu_inf;
            %If the assumptions about user behavior are correct, then fu_inf is the hidden fu.
            Feedback = [fu_inf,considered_kws];
            posterior = calculate_posterior(X_train, Y_train, Feedback, ...
                sparse_params, sparse_options);
        end
        
        
        %calculate training and test error
        Y_hat = X_test'*posterior.mean;
        Y_hat = Y_hat .* y_std + y_mean;
        Y_hat_train = X_train'*posterior.mean;
        Loss_1(method_num, :, user) = mean((Y_hat- Y_test).^2); %MSE
        Loss_2(method_num, :, user) = mean((Y_hat_train- Y_train).^2); %MSE on training in the normalized space
        
        
    end 
end
%we may want to investigate the inferred likelihood values later
save('FB_biased_inferred','FB_biased_inferred')
%% averaging and plotting
save('user_study_mse_results', 'Loss_1', 'Loss_2', 'sparse_options','sparse_params','Machine_estimates', 'best_alphas', ...
     'Method_list',  'num_features','num_trainingdata','RNG_SEED')
evaluate