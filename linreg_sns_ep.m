function [fa, si, converged, subfunctions] = linreg_sns_ep(y, x, pr, op, w_feedbacks, gamma_feedbacks, wsign_feedbacks, si)
% NOTE: this is a special version that takes gamma_feedbacks(:, 1) as
% probability of the feature being relevant (if kappa_prior = 0). If kappa
% prior is set, then it just uses gamma_feedbacks(:, 1) >= 0.5 as the feedback.

% -- Likelihood (y are data, f are feedbacks):
%    p(y_i|x_i,w,sigma2) = N(y_i|w'x_i, sigma2)
%    p(f_w_j|w_j,eta2) = N(f_w_j|w_j, eta2)
%    p(f_gamma_j|gamma_j) = I(gamma_j=1) Bernoulli(f_gamma_j|p_u) + I(gamma_j=0) Bernoulli(f_gamma_j|1-p_u)
% -- Prior:
%    p(w_j|gamma_j=1) = Normal(w_j|0, tau2)
%    p(w_j|gamma_j=0) = delta(w_j)
%    p(gamma_j|rho) = Bernoulli(gamma_j|rho)
%    p(rho) = Beta(rho|rho_a, rho_b)
%    p(sigma2^-1) = Gamma(sigma2^-1|sigma2_a,sigma2_b) or fixed sigma2
% -- Approximation;
%    q(w) = Normal(w|w.Mean, w_Var), w_Var = w.Tau^-1
%    q(gamma) = \prod_j Bernoulli(gamma_j|gamma.p_j)
%    q(sigma2^-1) = Gamma(sigma2^-1|sigma2_a,sigma2_b), mean: sigma2.imean
%    q(rho) = Beta(rho|rho.a,rho.b)
%
%    [TODO: add sign feedback modelling documentation]
%
%    sigma2 and rho are updated using VB (if not fixed), other terms using EP.
%
% Inputs:
% y                target values (n x 1)
% x                covariates (n x m)
% pr               prior and other fixed model parameters (struct)
% op               options for the EP algorithm (struct)
% w_feedbacks      values (1st column) and indices (2nd column) of feedback (n_w_feedbacks x 2)
% gamma_feedbacks  values (1st column, in (0,1)) and indices (2nd column) of feedback (n_gamma_feedbacks x 2)
% wsign_feedbacks  values (1st column, -1/1) and indices (2nd column) of feedback (n_wsf_feedbacks x 2)
% si               if given, (some of) site parameters initialized to these
%
% Outputs:
% fa         EP posterior approximation (struct)
% si         EP site terms (struct)
% converged  did EP converge or hit max_iter (1/0)
%
% Tomi Peltola, tomi.peltola@aalto.fi

if nargin < 5
    w_feedbacks = [];
end

if nargin < 6
    gamma_feedbacks = [];
end

if nargin < 7
    wsign_feedbacks = [];
end

[n, m] = size(x);
pr.n = n;
pr.m = m;
pr.yy = y' * y; % precompute
pr.xy = x' * y; % precompute
if op.degenerate_representation
    pr.x = x;
else
    pr.x = x;
    pr.xx = x' * x; % precompute
end
n_w_feedbacks = size(w_feedbacks, 1);
n_gamma_feedbacks = size(gamma_feedbacks, 1);
n_wsign_feedbacks = size(wsign_feedbacks, 1);

%% initialize (if si is given, prior sites are not re-initialized, but likelihood is)
if nargin < 8 || isempty(si)
    si.w_prior.w.normal_mu = zeros(m, 1);
    si.w_prior.w.normal_tau = (1 / pr.tau2) * ones(m, 1); % note: pr.tau2 is used here for initialization even if tau2 has prior (TODO: change this?)
    si.w_prior.gamma.bernoulli_p_nat = zeros(m, 1);
end
S_f = zeros(m, 1);
F_f = zeros(m, 1);
if n_w_feedbacks > 0
    for i = 1:n_w_feedbacks
        S_f(w_feedbacks(i, 2)) = 1;
        F_f(w_feedbacks(i, 2)) = w_feedbacks(i, 1);
    end
end
si.w_feedback.normal_Tau = (1 / pr.eta2) * S_f;
si.w_feedback.normal_Mu = (1 / pr.eta2) * F_f;
if isfield(pr, 'sigma2_prior') && pr.sigma2_prior
    si.y_lik.gamma_a = 0.5 * n;
    si.y_lik.gamma_b = 0.5 * pr.yy;
    sigma2_imean = (pr.sigma2_a + si.y_lik.gamma_a) / (pr.sigma2_b + si.y_lik.gamma_b);
    if op.degenerate_representation
        si.y_lik.normal_Tau_half = sqrt(sigma2_imean) * pr.x;
    else
        si.y_lik.normal_Tau = sigma2_imean * pr.xx;
    end
    si.y_lik.normal_Mu = sigma2_imean * pr.xy;
else
    if op.degenerate_representation
        si.y_lik.normal_Tau_half = (1 / sqrt(pr.sigma2)) * pr.x;
    else
        si.y_lik.normal_Tau = (1 / pr.sigma2) * pr.xx;
    end
    si.y_lik.normal_Mu = (1 / pr.sigma2) * pr.xy;
    pr.sigma2_prior = 0;
end
si.gamma_feedback.gamma.bernoulli_p_nat = zeros(m, 1);
si.gamma_feedback.p_u.bernoulli_p_nat = zeros(m, 1);

if isfield(pr, 'rho_prior') && pr.rho_prior
    rho_ = pr.rho_a / (pr.rho_a + pr.rho_b);
    si.gamma_prior.bernoulli_p_nat = log(rho_) - log1p(-rho_);
    si.gamma_prior.beta_a = zeros(m, 1);
    si.gamma_prior.beta_b = zeros(m, 1);
else
    si.gamma_prior.bernoulli_p_nat = log(pr.rho) - log1p(-pr.rho);
    pr.rho_prior = 0;
end

si.w_sign_feedback.bernoulli_p_nat = zeros(m, 1); % p_u
si.w_sign_feedback.normal_mu = zeros(m, 1);       % w
si.w_sign_feedback.normal_tau = zeros(m, 1);      % w

if isfield(pr, 'kappa_prior') && pr.kappa_prior
    %error('this version does not allow prior on the quality of user knowledge');
    if n_gamma_feedbacks > 0
     gamma_feedbacks(:, 1) = gamma_feedbacks(:, 1) >= 0.5;
    end
    p_u_ = pr.kappa_a / (pr.kappa_a + pr.kappa_b);
    si.w_sign_f_p_u_prior.bernoulli_p_nat = log(p_u_) - log1p(-p_u_);
    si.w_sign_f_p_u_prior.beta_a = zeros(m, 1);
    si.w_sign_f_p_u_prior.beta_b = zeros(m, 1);
    si.gamma_f_p_u_prior.bernoulli_p_nat = log(p_u_) - log1p(-p_u_);
    si.gamma_f_p_u_prior.beta_a = zeros(m, 1);
    si.gamma_f_p_u_prior.beta_b = zeros(m, 1);
else
    si.w_sign_f_p_u_prior.bernoulli_p_nat = log(pr.p_u) - log1p(-pr.p_u);
    si.gamma_f_p_u_prior.bernoulli_p_nat = (log(pr.p_u) - log1p(-pr.p_u)) * ones(m, 1);
    if n_gamma_feedbacks > 0
        %tmp_vals = max(1 - gamma_feedbacks(:, 1), gamma_feedbacks(:, 1));
        tmp_vals = min(max(1 - gamma_feedbacks(:, 1), gamma_feedbacks(:, 1)), 1-eps);        
        si.gamma_f_p_u_prior.bernoulli_p_nat(gamma_feedbacks(:, 2)) = log(tmp_vals) - log1p(-tmp_vals);
        gamma_feedbacks(:, 1) = gamma_feedbacks(:, 1) >= 0.5;
    end
    %pr.p_u_nat = log(pr.p_u) - log1p(-pr.p_u);
    pr.kappa_prior = 0;
end

if isfield(pr, 'tau2_prior') && pr.tau2_prior
    si.w_prior.tau2.normal_mu = zeros(m, 1);
    si.w_prior.tau2.normal_tau = zeros(m, 1);
    
    if ~isfield(op, 'hermite_x') % assume that if locations are given, weights will also be given 
        % Gauss-Hermite quadrature: using the weights and eval.locations from
        % EKF/UKF toolbox (http://becs.aalto.fi/en/research/bayes/ekfukf/)
        h_n = op.hermite_n;
        h_p = hermitepolynomial(h_n);
        op.hermite_x = roots(h_p);
        
        h_Wc = pow2(h_n-1) * factorial(h_n) * sqrt(pi) / h_n^2;
        h_p2 = hermitepolynomial(h_n - 1);
        op.hermite_W  = zeros(h_n, 1);
        for i = 1:h_n
            op.hermite_W(i) = h_Wc * polyval(h_p2, op.hermite_x(i)).^-2;
        end
        op.hermite_W = op.hermite_W / sqrt(pi);
    end
else
    pr.tau2_prior = 0;
end

% full approximation
fa = compute_full_approximation(si, pr, op);

% convergence diagnostics
conv.P_gamma_old = Inf * ones(m, 1);
conv.w_old = Inf * ones(m, 1);

update_inds = true(m, 1);

%% loop parallel EP
for iter = 1:op.max_iter
    %% w prior updates
    % cavity
    ca_w_prior = compute_sns_prior_cavity(fa, si.w_prior, op, pr);
    
    % moments of tilted dists
    [ti_w_prior, z_w] = compute_sns_prior_tilt(ca_w_prior, op, pr);
    
    % site updates
    si.w_prior = update_sns_prior_sites(si.w_prior, ca_w_prior, ti_w_prior, op, pr, update_inds);
    
    % full approx update
    fa = compute_full_approximation_w(fa, si, pr, op);
    fa = compute_full_approximation_gamma(fa, si, pr, op);
    if pr.tau2_prior
       fa = compute_full_approximation_tau2(fa, si, pr, op); 
    end

    %% gamma prior updates, EP for gamma, VB for rho
    if pr.rho_prior
        % VB
        si.gamma_prior = update_bernoulli_sites_vb(si.gamma_prior, fa.gamma.p, op);
        
        fa = compute_full_approximation_rho(fa, si, pr, op);
        
        % EP
        si.gamma_prior = update_bernoulli_sites_ep(si.gamma_prior, fa.gamma.p_nat, fa.rho.a, fa.rho.b, op);
       
        fa = compute_full_approximation_gamma(fa, si, pr, op);
    end

    %% sigma2 and (the associated) likelihood VB update
    if pr.sigma2_prior
        % sigma2 update
        si.y_lik = update_gaussian_lik_prec_site_vb(si.y_lik, fa.w, pr, op);
        
        fa = compute_full_approximation_sigma2(fa, si, pr, op);

        % likelihood update
        si.y_lik = update_gaussian_lik_normal_site_vb(si.y_lik, fa.sigma2.imean, pr, op);

        % full approx update
        fa = compute_full_approximation_w(fa, si, pr, op);
    end
    
    %% w sign feedback updates
    if n_wsign_feedbacks > 0
        % cavity
        ca_wsf = compute_wsf_cavity(fa, si.w_sign_feedback, op, pr);

        % moments of tilted dists
        ti_wsf = compute_wsf_tilt(ca_wsf, pr, wsign_feedbacks);

        % site updates
        si.w_sign_feedback = update_wsf_sites(si.w_sign_feedback, ca_wsf, ti_wsf, wsign_feedbacks, op, update_inds);

        %% full approx update (update only w part as only those sites have been updated)
        fa = compute_full_approximation_w(fa, si, pr, op);
        fa = compute_full_approximation_p_u(fa, si, pr, op);
    end
    
    %% gamma feedback updates
    if n_gamma_feedbacks > 0
        % cavity
        ca_gf = compute_gamma_f_lik_cavity(fa.gamma.p_nat, fa.gamma_f_p_u.p_nat, si.gamma_feedback, gamma_feedbacks(:, 2));

        % moments of tilted dists
        ti_gf = compute_gamma_f_lik_tilt(ca_gf, pr, gamma_feedbacks(:, 1));

        % site updates
        si.gamma_feedback = update_gamma_f_lik_sites(si.gamma_feedback, ca_gf, ti_gf, op, gamma_feedbacks(:, 2));

        % full approx update (update only gamma part as only those sites have been updated)
        fa = compute_full_approximation_gamma(fa, si, pr, op);
        fa = compute_full_approximation_p_u(fa, si, pr, op);
    end
    
    %% kappa prior updates, EP for p_u, VB for kappa
    if pr.kappa_prior
        % VB
        si.w_sign_f_p_u_prior = update_bernoulli_sites_vb(si.w_sign_f_p_u_prior, fa.w_sign_f_p_u.p, op);
        si.gamma_f_p_u_prior = update_bernoulli_sites_vb(si.gamma_f_p_u_prior, fa.gamma_f_p_u.p, op);
        
        fa = compute_full_approximation_kappa(fa, si, pr, op);
        
        % EP
        si.w_sign_f_p_u_prior = update_bernoulli_sites_ep(si.w_sign_f_p_u_prior, fa.w_sign_f_p_u.p_nat, fa.kappa.a, fa.kappa.b, op);
        si.gamma_f_p_u_prior = update_bernoulli_sites_ep(si.gamma_f_p_u_prior, fa.gamma_f_p_u.p_nat, fa.kappa.a, fa.kappa.b, op);

        fa = compute_full_approximation_p_u(fa, si, pr, op);
    end

    %% show progress and check for convergence
    [converged, conv] = report_progress_and_check_convergence(conv, iter, z_w, fa, op);
    if converged
        if op.verbosity > 0
            fprintf(1, 'EP converged on iteration %d\n', iter);
        end
        break
    end
    
    % note: taking & means that those that have dropped will not enter
    % updates again
    if op.degenerate_representation
        update_inds = update_inds & ~(abs(fa.w.Mean) < op.w_mean_update_threshold & (sum(fa.w.Tau_x_half .* fa.w.Tau_x_half)' + fa.w.Tau_diag) > op.w_prec_update_threshold);
    else
        update_inds = update_inds & ~(abs(fa.w.Mean) < op.w_mean_update_threshold & diag(fa.w.Tau) > op.w_prec_update_threshold);
    end
    
    %% update damp
    op.damp = op.damp * op.damp_decay;
end

if op.verbosity > 0 && converged == 0
    fprintf(1, 'EP hit maximum number of iterations\n');
end

if nargout > 3
    subfunctions.update_gaussian_lik_normal_site_vb = @update_gaussian_lik_normal_site_vb;
    subfunctions.update_gaussian_lik_prec_site_vb = @update_gaussian_lik_prec_site_vb;
    subfunctions.update_bernoulli_sites_vb = @update_bernoulli_sites_vb;
    subfunctions.update_bernoulli_sites_ep = @update_bernoulli_sites_ep;
    subfunctions.compute_gamma_f_lik_cavity = @compute_gamma_f_lik_cavity;
    subfunctions.compute_gamma_f_lik_tilt = @compute_gamma_f_lik_tilt;
    subfunctions.update_gamma_f_lik_sites = @update_gamma_f_lik_sites;
    subfunctions.compute_sns_prior_cavity = @compute_sns_prior_cavity;
    subfunctions.compute_sns_prior_tilt = @compute_sns_prior_tilt;
    subfunctions.update_sns_prior_sites = @update_sns_prior_sites;
    subfunctions.compute_full_approximation = @compute_full_approximation;
    subfunctions.compute_full_approximation_rho = @compute_full_approximation_rho;
    subfunctions.compute_full_approximation_sigma2 = @compute_full_approximation_sigma2;
    subfunctions.compute_full_approximation_w = @compute_full_approximation_w;
    subfunctions.compute_full_approximation_gamma = @compute_full_approximation_gamma;
    subfunctions.compute_wsf_tilt = @compute_wsf_tilt;
    subfunctions.compute_wsf_cavity = @compute_wsf_cavity;
    subfunctions.update_wsf_sites = @update_wsf_sites;
end

end


% TODO: Refactor: this is exactly the same as w cavity but without gamma.
function ca = compute_wsf_cavity(fa, si, op, pr)

m = pr.m;

if op.degenerate_representation
    %tmp = fa.w.degenerate_inner_chol \ (fa.w.Tau_x_half * diag(1 ./ fa.w.Tau_diag));
    tmp = fa.w.degenerate_inner_chol \ bsxfun(@times, fa.w.Tau_x_half, 1 ./ fa.w.Tau_diag');
    var_w = 1 ./ fa.w.Tau_diag - sum(tmp.^2)';
else
    tmp = fa.w.Tau_chol \ eye(m);
    var_w = sum(tmp.^2)';
end
    
denom = (1 - si.normal_tau .* var_w);
ca.normal_tau = denom ./ var_w;
ca.normal_mean = (fa.w.Mean - var_w .* si.normal_mu) ./ denom;
%assert(all(isfinite(ca.w.mean)))
%assert(all(isfinite(ca.w.tau)))

ca.bernoulli_p_nat = fa.w_sign_f_p_u.p_nat - si.bernoulli_p_nat;
ca.bernoulli_p = 1 ./ (1 + exp(-ca.bernoulli_p_nat));

end


function ti = compute_wsf_tilt(ca, pr, feedbacks)

% feedbacks: first is value, second index.
% Computes only those with feedback:
c_m = ca.normal_mean(feedbacks(:, 2));
c_s2 = 1 ./ ca.normal_tau(feedbacks(:, 2));
c_s = sqrt(c_s2);
ca_nr = -c_m ./ c_s;
n_dens = exp(-0.5 * ca_nr.^2) / sqrt(2 * pi);
%n_cdf = max(min(normcdf(ca_nr), 1-eps), eps); % TODO: could this cause problems?
n_cdf = normcdf(ca_nr); % z_m
%n_ccdf = 1 - n_cdf; % z_p
a = (feedbacks(:, 1) == -1) + feedbacks(:, 1) .* ca.bernoulli_p(feedbacks(:, 2));
% b = 1 - a;

zplus = a .* (1 - n_cdf);
zminus = (1 - a) .* n_cdf;
z = zplus + zminus;
% z = a * z_p + b * z_m = a * (1 - z_m) + (1 - a) * z_m = a - 2 * a .* z_m + z_m
%z = a - 2 * a .* n_cdf + n_cdf;

% note: a + b = 1, a - b = 2 * a - 1
ti.normal_mean = c_m + c_s .* n_dens .* (2 * a - 1) ./ z;
ti.normal_var = c_s2 + ti.normal_mean .* (c_m - ti.normal_mean);

ti.bernoulli_mean = ((feedbacks(:, 1) == -1) .* zminus + (feedbacks(:, 1) == 1) .* zplus) ./ z;
ti.bernoulli_mean = max(min(ti.bernoulli_mean, 1-eps), eps);

%assert(all(isfinite(ti.w.mean)))
%assert(all(isfinite(ti.w.var)))
end


% TODO: Refactor: this is the same as w prior updates but without gamma and computing only the ones with feedback.
function si = update_wsf_sites(si, ca, ti, feedbacks, op, update_inds)

% update only those with feedbacks
ca.normal_tau = ca.normal_tau(feedbacks(:, 2));
ca.normal_mean = ca.normal_mean(feedbacks(:, 2));

% skip negative cavs
if nargin < 6 || isempty(update_inds)
    update_inds = ca.normal_tau(:) > 0;
else
    update_inds = (ca.normal_tau(:) > 0) & update_inds(feedbacks(:, 2));
end

new_tau_w_site = 1 ./ ti.normal_var - ca.normal_tau;

switch op.robust_updates
    case 0
    case 1
        inds_tmp = new_tau_w_site(:) > 0;
        update_inds = update_inds & inds_tmp;
    case 2
        inds = new_tau_w_site(:) <= 0;
        new_tau_w_site(inds) = op.min_site_prec;
        ti.normal_var(inds) = 1./(op.min_site_prec + ca.normal_tau(inds));
        
        inds = new_tau_w_site(:) > op.max_site_prec;
        new_tau_w_site(inds) = op.max_site_prec;
        ti.normal_var(inds) = 1./(op.max_site_prec + ca.normal_tau(inds));
end
new_mu_w_site = ti.normal_mean ./ ti.normal_var - ca.normal_tau .* ca.normal_mean;
inds = feedbacks(update_inds, 2);
si.normal_tau(inds) = (1 - op.damp) * si.normal_tau(inds) + op.damp * new_tau_w_site(update_inds);
si.normal_mu(inds) = (1 - op.damp) * si.normal_mu(inds) + op.damp * new_mu_w_site(update_inds);

si.bernoulli_p_nat(inds) = (1 - op.damp) * si.bernoulli_p_nat(inds) + op.damp * (log(ti.bernoulli_mean(update_inds)) - log1p(-ti.bernoulli_mean(update_inds)) - ca.bernoulli_p_nat(update_inds));

end


function si = update_gaussian_lik_normal_site_vb(si, prec_mean, pr, op)

if op.degenerate_representation
    si.normal_Tau_half = sqrt(prec_mean) * pr.x;
else
    si.normal_Tau = prec_mean * pr.xx;
end
si.normal_Mu = prec_mean * pr.xy;

end


function si = update_gaussian_lik_prec_site_vb(si, w, pr, op)

if op.degenerate_representation
    %tr_tmp1 = pr.x * diag(1 ./ w.Tau_diag);
    tr_tmp1a = bsxfun(@times, pr.x, 1 ./ w.Tau_diag');
    tr_tmp1 = tr_tmp1a(:)' * pr.x(:);
    
    %tr_tmp2 = w.degenerate_inner_chol \ (w.Tau_x_half * diag(1 ./ w.Tau_diag) * pr.x');
    tr_tmp2 = w.degenerate_inner_chol \ (w.Tau_x_half * tr_tmp1a');
    tr_tmp2 = tr_tmp2(:)' * tr_tmp2(:);
    
    mx = pr.x * w.Mean;
    si.gamma_b = (1 - op.damp) * si.gamma_b + op.damp * (0.5 * (pr.yy - 2 * (w.Mean' * pr.xy) + tr_tmp1 - tr_tmp2 + mx' * mx));
else
    tr_tmp = pr.x / w.Tau_chol';
    
    si.gamma_b = (1 - op.damp) * si.gamma_b + op.damp * (0.5 * (pr.yy - 2 * (w.Mean' * pr.xy) + tr_tmp(:)' * tr_tmp(:) + w.Mean' * pr.xx * w.Mean));
    %si.lik.sigma2.b = 0.5 * (pr.yy - 2 * (fa.w.Mean' * pr.xy) + tr_tmp(:)' * tr_tmp(:) + fa.w.Mean' * pr.xx * fa.w.Mean);
end

end


function si = update_bernoulli_sites_vb(si, p, op)
% This updates the conditioning variable (probability parameter).

si.beta_a = (1 - op.damp) * si.beta_a + op.damp * p;
si.beta_b = (1 - op.damp) * si.beta_b + op.damp * (1 - p);
%si.prior.rho.a = fa.gamma.p;
%si.prior.rho.b = (1 - fa.gamma.p);

end


function si = update_bernoulli_sites_ep(si, fa_bernoulli_p_nat, fa_beta_a, fa_beta_b, op)
% This updates the main variable (indicator variable).

% cavity
cav_nat = fa_bernoulli_p_nat - si.bernoulli_p_nat;
cav_a_m_cav_nat = (fa_beta_a - si.beta_a - 1 + eps) .* exp(cav_nat);
cav_b = fa_beta_b - si.beta_b - 1 + eps;

% tilt
ti_mean = cav_a_m_cav_nat ./ (cav_a_m_cav_nat + cav_b);
ti_mean = max(min(ti_mean, 1-eps), eps);

% site update
si.bernoulli_p_nat = (1 - op.damp) * si.bernoulli_p_nat + op.damp * (log(ti_mean) - log1p(-ti_mean) - cav_nat);

end


function ca = compute_gamma_f_lik_cavity(gamma_bernoulli_p_nat, p_u_bernoulli_p_nat, si, inds)

if nargin < 4
    ca.gamma_bernoulli_p_nat = gamma_bernoulli_p_nat - si.gamma.bernoulli_p_nat;
    ca.p_u_bernoulli_p_nat = p_u_bernoulli_p_nat - si.p_u.bernoulli_p_nat;
else
    ca.gamma_bernoulli_p_nat = gamma_bernoulli_p_nat(inds) - si.gamma.bernoulli_p_nat(inds);
    ca.p_u_bernoulli_p_nat = p_u_bernoulli_p_nat(inds) - si.p_u.bernoulli_p_nat(inds);
end

end


function ti = compute_gamma_f_lik_tilt(ca, pr, observations)

%ti.bernoulli_mean = 1 ./ (1 + exp(-(ca.bernoulli_p_nat + (2 * observations - 1) .* pr.p_u_nat)));
%ti.bernoulli_mean = max(min(ti.bernoulli_mean, 1-eps), eps);
ti.gamma_bernoulli_mean = 1 ./ (1 + exp(-(ca.gamma_bernoulli_p_nat + (2 * observations - 1) .* ca.p_u_bernoulli_p_nat)));
ti.gamma_bernoulli_mean = max(min(ti.gamma_bernoulli_mean, 1-eps), eps);

ti.p_u_bernoulli_mean = 1 ./ (1 + exp(-(ca.p_u_bernoulli_p_nat + (2 * observations - 1) .* ca.gamma_bernoulli_p_nat)));
ti.p_u_bernoulli_mean = max(min(ti.p_u_bernoulli_mean, 1-eps), eps);

end


function si = update_gamma_f_lik_sites(si, ca, ti, op, inds)

if nargin < 5
    si.gamma.bernoulli_p_nat = (1 - op.damp) * si.gamma.bernoulli_p_nat + op.damp * (log(ti.gamma_bernoulli_mean) - log1p(-ti.gamma_bernoulli_mean) - ca.gamma_bernoulli_p_nat);
    si.p_u.bernoulli_p_nat = (1 - op.damp) * si.p_u.bernoulli_p_nat + op.damp * (log(ti.p_u_bernoulli_mean) - log1p(-ti.p_u_bernoulli_mean) - ca.p_u_bernoulli_p_nat);
else
    si.gamma.bernoulli_p_nat(inds) = (1 - op.damp) * si.gamma.bernoulli_p_nat(inds) + op.damp * (log(ti.gamma_bernoulli_mean) - log1p(-ti.gamma_bernoulli_mean) - ca.gamma_bernoulli_p_nat);
    si.p_u.bernoulli_p_nat(inds) = (1 - op.damp) * si.p_u.bernoulli_p_nat(inds) + op.damp * (log(ti.p_u_bernoulli_mean) - log1p(-ti.p_u_bernoulli_mean) - ca.p_u_bernoulli_p_nat);
end

end


function ca = compute_sns_prior_cavity(fa, si, op, pr)

m = pr.m;

if op.degenerate_representation
    %tmp = fa.w.degenerate_inner_chol \ (fa.w.Tau_x_half * diag(1 ./ fa.w.Tau_diag));
    tmp = fa.w.degenerate_inner_chol \ bsxfun(@times, fa.w.Tau_x_half, 1 ./ fa.w.Tau_diag');
    var_w = 1 ./ fa.w.Tau_diag - sum(tmp.^2)';
else
    tmp = fa.w.Tau_chol \ eye(m);
    var_w = sum(tmp.^2)';
end

denom = (1 - si.w.normal_tau .* var_w);
ca.w.normal_tau = denom ./ var_w;
ca.w.normal_mean = (fa.w.Mean - var_w .* si.w.normal_mu) ./ denom;

ca.gamma.bernoulli_p_nat = fa.gamma.p_nat - si.gamma.bernoulli_p_nat;
ca.gamma.bernoulli_p = 1 ./ (1 + exp(-ca.gamma.bernoulli_p_nat));

if pr.tau2_prior
    ca.tau2.normal_tau = fa.tau2.tau - si.tau2.normal_tau;
    ca.tau2.normal_var = 1 ./ ca.tau2.normal_tau;
    ca.tau2.normal_mu = fa.tau2.mu - si.tau2.normal_mu;
    ca.tau2.normal_mean = ca.tau2.normal_mu .* ca.tau2.normal_var;
end

end


function [ti, z] = compute_sns_prior_tilt(ca, op, pr)

if pr.tau2_prior
    % todo:
    % -integrate over tau2 for computing the moments of gamma and w
    % -compute the moments of tau2
    % -actually, put log-normal prior on the scale rather than variance?
    % (if yes, change name to pr.tau_prior?)
    % make the code better and clearer (e.g., now z and zz is computed)

    g_var = 1 ./ ca.w.normal_tau; % for gamma0
    mcav2 = ca.w.normal_mean.^2;
    log_z_gamma0 = log1p(-ca.gamma.bernoulli_p) - 0.5 * log(g_var) - 0.5 * mcav2 ./ g_var;  
 
    et = bsxfun(@plus, op.hermite_x * sqrt(2 * ca.tau2.normal_var)', ca.tau2.normal_mean'); % sqrt(2) comes from Gauss-Hermite change of variables
    tau2 = exp(2 * et);
    g_var = bsxfun(@plus, tau2, g_var'); % for gamma1
    log_z_gamma1_nop = -0.5 * log(g_var) - bsxfun(@rdivide, 0.5 * mcav2', g_var);
    log_z_gamma1_max = max(log_z_gamma1_nop);
    z_gamma1_nop = exp(bsxfun(@minus, log_z_gamma1_nop, log_z_gamma1_max));
    log_z_gamma1 = log(ca.gamma.bernoulli_p) + (log(op.hermite_W' * z_gamma1_nop) + log_z_gamma1_max)';
    
    z_gamma0 = exp(log_z_gamma0 - log_z_gamma1);
    z_gamma1 = ones(size(log_z_gamma1));
    z = 1 + z_gamma0;
    
    zz = exp(log_z_gamma0) + exp(log_z_gamma1); % TODO: fix this; no need to compute z twice with different scalings...
    ti.tau2.normal_mean = (ca.gamma.bernoulli_p .* ((op.hermite_W' * (et .* z_gamma1_nop)) .* exp(log_z_gamma1_max))' + exp(log_z_gamma0) .* ca.tau2.normal_mean) ./ zz;
    tau2_e2 = (ca.gamma.bernoulli_p .* ((op.hermite_W' * (et.^2 .* z_gamma1_nop)) .* exp(log_z_gamma1_max))' + exp(log_z_gamma0) .* (ca.tau2.normal_mean.^2 + ca.tau2.normal_var)) ./ zz;
    ti.tau2.normal_var = tau2_e2 - ti.tau2.normal_mean.^2;
    
    t = bsxfun(@plus, ca.w.normal_tau', 1 ./ tau2);
    ti.w.normal_mean = (ca.gamma.bernoulli_p .* ca.w.normal_tau .* ca.w.normal_mean) .* ((op.hermite_W' * (z_gamma1_nop ./ t)) .* exp(log_z_gamma1_max))' ./ zz;
    %ti.w.normal_mean = z_gamma1 .* (ca.w.normal_tau .* ca.w.normal_mean) ./ t ./ z;
    ti_normal_e2 = (ca.gamma.bernoulli_p .* (((op.hermite_W' * (z_gamma1_nop ./ t)) .* exp(log_z_gamma1_max))' +  (ca.w.normal_tau .* ca.w.normal_mean).^2 .* ((op.hermite_W' * (z_gamma1_nop ./ t.^2)) .* exp(log_z_gamma1_max))')) ./ zz;
    %ti_normal_e2 = z_gamma1 .* (1 ./ t + 1 ./ t.^2 .* (ca.w.normal_tau .* ca.w.normal_mean).^2) ./ z;
    ti.w.normal_var = ti_normal_e2 - ti.w.normal_mean.^2;

    ti.gamma.bernoulli_mean = z_gamma1 ./ z;
    ti.gamma.bernoulli_mean = max(min(ti.gamma.bernoulli_mean, 1-eps), eps);
else
    t = ca.w.normal_tau + 1 ./ pr.tau2;
    
    g_var = 1 ./ ca.w.normal_tau; % for gamma0
    mcav2 = ca.w.normal_mean.^2;
    log_z_gamma0 = log1p(-ca.gamma.bernoulli_p) - 0.5 * log(g_var) - 0.5 * mcav2 ./ g_var;
    g_var = pr.tau2 + g_var; % for gamma1
    log_z_gamma1 = log(ca.gamma.bernoulli_p) - 0.5 * log(g_var) - 0.5 * mcav2 ./ g_var;
    z_gamma0 = exp(log_z_gamma0 - log_z_gamma1);
    z_gamma1 = ones(size(log_z_gamma1));
    z = 1 + z_gamma0;
    
    ti.w.normal_mean = z_gamma1 .* (ca.w.normal_tau .* ca.w.normal_mean) ./ t ./ z;
    ti_normal_e2 = z_gamma1 .* (1 ./ t + 1 ./ t.^2 .* (ca.w.normal_tau .* ca.w.normal_mean).^2) ./ z;
    ti.w.normal_var = ti_normal_e2 - ti.w.normal_mean.^2;
    
    ti.gamma.bernoulli_mean = z_gamma1 ./ z;
    ti.gamma.bernoulli_mean = max(min(ti.gamma.bernoulli_mean, 1-eps), eps);
end

end


function si = update_sns_prior_sites(si, ca, ti, op, pr, update_inds)

% skip negative cavs
if nargin < 6 || isempty(update_inds)
    update_inds = ca.w.normal_tau(:) > 0;
else
    update_inds = (ca.w.normal_tau(:) > 0) & update_inds;
end

new_tau_w_site = 1 ./ ti.w.normal_var - ca.w.normal_tau;

switch op.robust_updates
    case 0
    case 1
        inds_tmp = new_tau_w_site(:) > 0;
        update_inds = update_inds & inds_tmp;
    case 2
        inds = new_tau_w_site(:) <= 0;
        new_tau_w_site(inds) = op.min_site_prec;
        ti.w.normal_var(inds) = 1./(op.min_site_prec + ca.w.normal_tau(inds));
end
new_mu_w_site = ti.w.normal_mean ./ ti.w.normal_var - ca.w.normal_tau .* ca.w.normal_mean;
si.w.normal_tau(update_inds) = (1 - op.damp) * si.w.normal_tau(update_inds) + op.damp * new_tau_w_site(update_inds);
si.w.normal_mu(update_inds) = (1 - op.damp) * si.w.normal_mu(update_inds) + op.damp * new_mu_w_site(update_inds);

si.gamma.bernoulli_p_nat(update_inds) = (1 - op.damp) * si.gamma.bernoulli_p_nat(update_inds) + op.damp * (log(ti.gamma.bernoulli_mean(update_inds)) - log1p(-ti.gamma.bernoulli_mean(update_inds)) - ca.gamma.bernoulli_p_nat(update_inds));

if pr.tau2_prior
    % TODO: need to worry about negative cavities and/or negative site
    % variances for these also?
    si.tau2.normal_tau(update_inds) = (1 - op.damp) * si.tau2.normal_tau(update_inds) + op.damp * (1 ./ ti.tau2.normal_var(update_inds) - ca.tau2.normal_tau(update_inds));
    si.tau2.normal_mu(update_inds) = (1 - op.damp) * si.tau2.normal_mu(update_inds) + op.damp * (ti.tau2.normal_mean(update_inds) ./ ti.tau2.normal_var(update_inds) - ca.tau2.normal_mu(update_inds));
end

end


function fa = compute_full_approximation(si, pr, op)

fa = struct;
fa = compute_full_approximation_w(fa, si, pr, op);
fa = compute_full_approximation_gamma(fa, si, pr, op);
fa = compute_full_approximation_p_u(fa, si, pr, op);
if pr.sigma2_prior
    fa = compute_full_approximation_sigma2(fa, si, pr, op);
end
if pr.rho_prior
    fa = compute_full_approximation_rho(fa, si, pr, op);
end
if pr.kappa_prior
    fa = compute_full_approximation_kappa(fa, si, pr, op);
end
if pr.tau2_prior
    fa = compute_full_approximation_tau2(fa, si, pr, op);
end

end


function fa = compute_full_approximation_tau2(fa, si, pr, op)

if pr.tau2_shared
  fa.tau2.mu = pr.tau2_mu + sum(si.w_prior.tau2.normal_mu);
  fa.tau2.tau = pr.tau2_tau + sum(si.w_prior.tau2.normal_tau);
else
  fa.tau2.mu = pr.tau2_mu + si.w_prior.tau2.normal_mu;
  fa.tau2.tau = pr.tau2_tau + si.w_prior.tau2.normal_tau;
end

end


function fa = compute_full_approximation_rho(fa, si, pr, op)

% These are Beta distribution parameters in the common parametrization;
% pr params are also, while si params are natural parameters.
fa.rho.a = sum(si.gamma_prior.beta_a) + pr.rho_a;
fa.rho.b = sum(si.gamma_prior.beta_b) + pr.rho_b;

end


function fa = compute_full_approximation_kappa(fa, si, pr, op)

% These are Beta distribution parameters in the common parametrization;
% pr params are also, while si params are natural parameters.
fa.kappa.a = sum(si.w_sign_f_p_u_prior.beta_a) + sum(si.gamma_f_p_u_prior.beta_a) + pr.kappa_a;
fa.kappa.b = sum(si.w_sign_f_p_u_prior.beta_b) + sum(si.gamma_f_p_u_prior.beta_b) + pr.kappa_b;

end


function fa = compute_full_approximation_sigma2(fa, si, pr, op)

% a and b are in the common parametrization of Gamma (the one with mean = a/b)
fa.sigma2.imean = (pr.sigma2_a + si.y_lik.gamma_a) / (pr.sigma2_b + si.y_lik.gamma_b); % note: approx is for sigma2^-1

end


function fa = compute_full_approximation_w(fa, si, pr, op)

% TODO: get rid of diag-calls for faster alternatives
% m x m and m x 1
if op.degenerate_representation
    fa.w.Tau_x_half = si.y_lik.normal_Tau_half;
    fa.w.Tau_diag = si.w_feedback.normal_Tau + si.w_prior.w.normal_tau + si.w_sign_feedback.normal_tau;
    fa.w.Mu = si.y_lik.normal_Mu + si.w_feedback.normal_Mu + si.w_prior.w.normal_mu + si.w_sign_feedback.normal_mu;
    %inner = fa.w.Tau_x_half * diag(1 ./ fa.w.Tau_diag) * fa.w.Tau_x_half';
    inner = bsxfun(@times, fa.w.Tau_x_half, 1 ./ fa.w.Tau_diag') * fa.w.Tau_x_half';
    inner(1:(size(inner, 1)+1):end) = inner(1:(size(inner, 1)+1):end) + 1; 
    fa.w.degenerate_inner_chol = chol(inner, 'lower');
    dmu = fa.w.Mu ./ fa.w.Tau_diag;
    %fa.w.Mean = dmu - diag(1 ./ fa.w.Tau_diag) * (fa.w.Tau_x_half' * (fa.w.degenerate_inner_chol' \ (fa.w.degenerate_inner_chol \ (fa.w.Tau_x_half * dmu))));
    fa.w.Mean = dmu - bsxfun(@times, 1 ./ fa.w.Tau_diag, fa.w.Tau_x_half' * (fa.w.degenerate_inner_chol' \ (fa.w.degenerate_inner_chol \ (fa.w.Tau_x_half * dmu))));
else
    fa.w.Tau = si.y_lik.normal_Tau + diag(si.w_feedback.normal_Tau) + diag(si.w_prior.w.normal_tau) + diag(si.w_sign_feedback.normal_tau);
    fa.w.Tau_chol = chol(fa.w.Tau, 'lower');
    fa.w.Mu = si.y_lik.normal_Mu + si.w_feedback.normal_Mu + si.w_prior.w.normal_mu + si.w_sign_feedback.normal_mu;
    fa.w.Mean = fa.w.Tau_chol' \ (fa.w.Tau_chol \ fa.w.Mu);
end
    
end


function fa = compute_full_approximation_gamma(fa, si, pr, op)

fa.gamma.p_nat = si.w_prior.gamma.bernoulli_p_nat + si.gamma_feedback.gamma.bernoulli_p_nat + si.gamma_prior.bernoulli_p_nat;
fa.gamma.p = 1 ./ (1 + exp(-fa.gamma.p_nat));

end


function fa = compute_full_approximation_p_u(fa, si, pr, op)

fa.w_sign_f_p_u.p_nat = si.w_sign_feedback.bernoulli_p_nat + si.w_sign_f_p_u_prior.bernoulli_p_nat;
fa.w_sign_f_p_u.p = 1 ./ (1 + exp(-fa.w_sign_f_p_u.p_nat));

fa.gamma_f_p_u.p_nat = si.gamma_feedback.p_u.bernoulli_p_nat + si.gamma_f_p_u_prior.bernoulli_p_nat;
fa.gamma_f_p_u.p = 1 ./ (1 + exp(-fa.gamma_f_p_u.p_nat));

end


function [converged, conv] = report_progress_and_check_convergence(conv, iter, z, fa, op)

conv_P_gamma = mean(abs(fa.gamma.p(:) - conv.P_gamma_old(:)));
conv_w = mean(abs(fa.w.Mean(:) - conv.w_old(:)));

if op.verbosity > 0 && mod(iter, op.verbosity) == 0
    fprintf(1, '%d, conv = [%.2e %.2e], damp = %.2e\n', iter, conv_w, conv_P_gamma, op.damp);
end

%converged = conv_z < op.threshold && conv_P_gamma < op.threshold;
converged = conv_P_gamma < op.gamma_threshold && conv_w < op.w_threshold;

conv.w_old = fa.w.Mean;
conv.P_gamma_old = fa.gamma.p;

end
