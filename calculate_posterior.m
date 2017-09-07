function [ posterior ] = calculate_posterior(X, Y, feedback, sparse_params, sparse_options)
% Calculate the posterior distribution given the observed data and user feedback
% Inputs:
% X             covariates (d x n)
% Y             response values
% feedback      values (1st column) and indices (2nd column) of feedback (n_feedbacks x 2)

    %assume sparse prior (spike and slab) and approximate the posterior with EP
    %weights are approximated by a multivariate Gaussian distribution.
    %latent variables are approximated by Bernoulli distribution.
    if ~isempty(feedback)
        %remove the feedback that the user said "don't know"
        dont_know_fb = feedback(:,1)==-1;
        feedback(dont_know_fb,:) = [];
    end
    [fa, si, converged, subfuncs] = linreg_sns_ep(Y, X', sparse_params, sparse_options, [] , feedback, sparse_options.si);
    if converged ~= 1
        disp('linreg_sns_ep did not converge')
    end
    posterior.si = si;
    posterior.fa = fa;
    %posterior.sigma = inv(fa.w.Tau);
    posterior.sigma = fa.w.Tau_chol' \ (fa.w.Tau_chol \ eye(size(fa.w.Tau_chol))); % this should be faster than inv?
    posterior.mean  = fa.w.Mean;
    posterior.p     = fa.gamma.p;
    posterior.ep_subfunctions = subfuncs;

end