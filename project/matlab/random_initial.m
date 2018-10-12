% random initial

close all;
clear;clc
O = 3;  % 3d position 
T = 400;
nex = 3; % eg. 3 person's data 
M = 3;  % state# of a single joint
Q = 3;  % state# of activity 
cov_type = 'full';

% load the data
data = randn(O,T,nex);
load person_1.mat
data(:,:,1)=still(1:T,1:3)';

load person_2.mat
data(:,:,2)=still(1:T,1:3)';

load person_3.mat
data(:,:,3)=still(1:T,1:3)';

% initial guess of parameters
p0 = normalise(rand(Q,1));
A = mk_stochastic(rand(Q,Q));

[mu0, Sigma0] = mixgauss_init(Q*M, data, cov_type);
mu0 = reshape(mu0, [O Q M]);
Sigma0 = reshape(Sigma0, [O O Q M]);
B = mk_stochastic(rand(Q,M));

[LL, p_est, A_est, mu_est, Sigma_est, B_est] = ...
    mhmm_em(data, p0, A, mu0, Sigma0, B, 'max_iter', 50);


loglik = mhmm_logprob(data, p_est, A_est, mu_est, Sigma_est, B_est)  %
% compute the log-likelihood 

% test another activity of person from 1 to 3
% load person_3.mat
data2=drinking_water(1:T,1:3)';
loglik = mhmm_logprob(data2, p_est, A_est, mu_est, Sigma_est, B_est)

