% This script is designed to run the overfitting experiments based on
% gathered feedbacks from the user study. The feedbacks are on probability 
% of relevance of features. 

close all
clear all
rng(54321)
RNG_SEED = rng;

%% Load Amazon dataset and user feedbacks from the user

data_addr = 'Data-Exp1\'; %data directory
% load 'X_train', 'Y_train', 'X_test', 'Y_test','y_mean','y_std'
% X and Y for training data are in the normalized space (zero mean unit std)
% X test is normalized based on training data and Y test is unnormalzied. 
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

%Features that were used in the user study
considered_kws = setdiff(1:num_features,I_dont_know_indices)';

%model parameters
% The model parameters are set based on:
% Daee, P., Peltola, T., Soare, M. et al. Mach Learn (2017) 106: 1599. 
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
              'p_u', 0.99, ...  %(NOT USED HERE)
              'rho_prior', 0, ...
              'rho', 0.3, ...
              'tau2_prior', 0, ...
              'tau2', 0.1^2, ...
              'eta2', 0.1^2);   %(NOT USED HERE)
          
% Put distribution assumptions on sigma2 
sparse_params.sigma2_prior  = 1;
sparse_params.sigma2_a  = 1;
sparse_params.sigma2_b  = 1;

%% METHOD TO COMPARE

if select_bias_experiment
    %for the biased experiment compare the following conditions
    Method_list = {'no feedback','User FB before correction', 'User FB after correction'};
else
    %for the baseline experiment biased correction is NA since the users did not see the machine estimates
    Method_list = {'no feedback','User FB before correction'};
    
end
%number of methods that we want to consider
num_methods = size(Method_list,2); 

%% Main algorithm
MSE_test = zeros(num_methods, num_users); % MSE on test
MSE_train = zeros(num_methods, num_users); % MSE on train

%Calculate the posterior before observing any feedbacks
posterior_no_fb = calculate_posterior(X_train, Y_train, [], ...
     sparse_params, sparse_options);
% compute the machine estimates that were shown to the users
Machine_estimates = posterior_no_fb.p(considered_kws);
Machine_estimates = round(100*Machine_estimates)/100; 

%This later will be used in the word_analysis script
FB_biased_inferred = zeros(size(FB_source,1), num_users); 

tic
for user = 1:num_users
    disp(['user number ', num2str(user), ' from ', num2str(num_users), '. acc time = ', num2str(toc) ]);
    for method_num = 1:num_methods
        method_name = Method_list(method_num);

        %Feedback = values (1st column) and indices (2nd column) of user feedback
        Feedback = [];            %only used in experimental design methdos        
        sparse_options.si = [];   %no need to carry prior site terms between interactions
        
        %% Based on the selected method, compute the proper posterior and calculate the errors
        if find(strcmp('no feedback', method_name))
            posterior = posterior_no_fb;
        end
        
        if find(strcmp('User FB before correction', method_name))
            %directly use user feedback in the posterior
            Feedback = [Feedback_all(:,user),considered_kws];
            posterior = calculate_posterior(X_train, Y_train, Feedback, ...
                sparse_params, sparse_options);
        end
               
        if find(strcmp('User FB after correction', method_name))
            %perform the user model correction and then use user feedback in the posterior
            tr_p = posterior_no_fb.p(considered_kws);
            % user's update knowledge based on marginal inclusion probabilities         
            fu_upd = Feedback_all(:,user); %user updated (biased) feedback
            I_dont_knows = fu_upd == -1;
            %Infer user hidden likelihood:
            % fu_inf = f (1 - p) / (f (1 - p) + (1 - f) p)
            % f is the given feedback (fu_upd), p is the posterior of training data (tr_p)
            numerator = fu_upd .* (1-tr_p);
            denominator = numerator + (1-fu_upd) .* tr_p;
            fu_inf = numerator./denominator;
            % exclude those that the user said "I don't know"
            fu_inf(I_dont_knows) = -1;
            FB_biased_inferred(:,user) = fu_inf;
            %If the assumptions about user behavior are correct, then fu_inf is the hidden fu.
            Feedback_inferred = [fu_inf,considered_kws];
            posterior = calculate_posterior(X_train, Y_train, Feedback_inferred, ...
                sparse_params, sparse_options);
        end
        
        %calculate training and test error
        Y_hat = X_test'*posterior.mean;
        Y_hat = Y_hat .* y_std + y_mean;
        Y_hat_train = X_train'*posterior.mean;
        MSE_test(method_num, user) = mean((Y_hat- Y_test).^2); %MSE
        MSE_train(method_num, user) = mean((Y_hat_train- Y_train).^2); %MSE on training in the normalized space
    end       
end
%we may want to investigate the inferred likelihood values later
save('FB_biased_inferred','FB_biased_inferred')
%% averaging and plotting
save('user_study_mse_results', 'MSE_test', 'MSE_train', 'sparse_options','sparse_params','Machine_estimates', ...
     'Method_list',  'num_features','num_trainingdata','RNG_SEED')
evaluate