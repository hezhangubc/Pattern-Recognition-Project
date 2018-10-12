% Matlab code for EECE 562 project
% This code is to implement human activity recognition 
%
% notes: 
%      1. For table 1, use case1 and respectively change Itest from [1] to
%      [2],[3],[4], and [1 2 3 4]
%      2. Uncomment case 2 instead, and test unseen case 
%      3. change node from 1 to 11 to try different skeletons

close all; 
clear;clc

%% para. setting
Ifull=[1 2 3 4];   % full dataset of person 1 to 4

%## case 1 START ## 
Itrain=[ 2 3 4 ];  % full dataset used for training,repace it with other set if necessary
Itest=[1];        % change this value respectively to 2,3,4
%## case 1 END ##

% %## case 2 START ##
% Itrain=[2 3 4];  % full dataset used for training,repace it with other set if necessary
% Itest=[1]; 
% %## case 2 END ##

nSamples=200;       % # of random samples to test

C={'still','talking_on_the_phone','writing_on_the_whiteboard','drinking_water',...
    'rinsing_mouth_with_water','brushing_teeth','wearing_contact_lenses_2','talking_on_couch',...
    'relaxing_on_couch','cooking_chopping','cooking_stirring','opening_pill_container','working_on_computer'};

T=200;             % length of time slices, first half for training, seond for testing
O = 3;  % 3d position 
nex = length(Itrain); % eg. 3 person's data 
M = 5;  % state# of a single joint
Q = 3;  % state# of activity  % change  Q to other number
node=1; % skeleton node
cov_type = 'full';
Data=zeros(3*length(C),T,length(Ifull));  
A_all=zeros(Q,Q,length(C));   % prob. trans. mat. of each activity
B_all=zeros(Q,M,length(C));   % obser. prob. trans. mat.
p_all=zeros(Q,length(C));
mu_all=zeros(3,Q,M,length(C));
Sigma_all=zeros(3,3,Q,M,length(C));

loglik=zeros(1,length(C));
label_true=(1:length(C))'*ones(1,nSamples);  % true label
label_est=zeros(length(C),nSamples);   % recognitized label 

%% training GMM-HMM model for each activity
% load and transform data for easier manipulation
for i=1:length(Ifull)
    load (['person_',num2str(i),'.mat']);
    for j=1:length(C)
        Dtmp=eval(cell2mat(C(j)));     % Dtmp: tempory variable, transform cell-named data to variable
        Data((j-1)*3+1:j*3,:,i)=Dtmp(1:T,1:3)';
    end
end

% train GMM-HMM model 
for j=1:length(C)
    clc;
% initial guess of parameters
    p0 = normalise(rand(Q,1));
    A = mk_stochastic(rand(Q,Q));
    
    data=Data((j-1)*3+1:j*3,1:floor(T/2),Itrain);
    [mu0, Sigma0] = mixgauss_init(Q*M, data, cov_type);
    mu0 = reshape(mu0, [O Q M]);
    Sigma0 = reshape(Sigma0, [O O Q M]);
    B = mk_stochastic(rand(Q,M));

 % train the model using EM
    [LL, p_est, A_est, mu_est, Sigma_est, B_est] = ...
        mhmm_em(data, p0, A, mu0, Sigma0, B, 'max_iter', 50);
 % save para.
    A_all(:,:,j)=A_est;
    B_all(:,:,j)=B_est;
    p_all(:,j)=p_est;
    mu_all(:,:,:,j)=mu_est;
    Sigma_all(:,:,:,:,j)=Sigma_est;
end

  fprintf('recognization in process, please wait: \n');
 %% testing stage -- random selected samples (unseen data)
 for j=1:length(C)
  % generate random testing subject
     subj_idx=randi(length(Itest),1,nSamples);  
  % generate random starting point given subject
     start_idx=randi(floor(T/4),1,nSamples);   % for the 2nd half T/2, using slideing window of .5 overlapping
  % generate random testing samples
     for k=1:nSamples
         data=Data((j-1)*3+1:j*3,floor(T/2)+start_idx(k):floor(T/2)+start_idx(k)+floor(T/4)-1,Itest(subj_idx(k)));
         for kk=1:length(C)                    % matching over all activities
             A_tmp=reshape(A_all(:,:,kk),size(A_est));
             B_tmp=reshape(B_all(:,:,kk),size(B_est));
             p_tmp=p_all(:,kk);
             mu_tmp=reshape(mu_all(:,:,:,kk),size(mu_est));
             Sigma_tmp=reshape(Sigma_all(:,:,:,:,kk),size(Sigma_est));
             loglik(kk) = mhmm_logprob(data, p_tmp, A_tmp, mu_tmp, Sigma_tmp, B_tmp);
         end
         
         [~,label_est(j,k)]=max(loglik);   % recognized activity 
         
     end
     
 end
 
 % get accuracy
   accuracy=mean(label_est==label_true,2);  
   clc;
   fprintf('recognition accuracy is: \n');
   disp(accuracy);
   fprintf('overall average accuracy is: \n');
   disp(mean(accuracy));
 
   % %record accuracy results  (for random tests) 
   % result for case 1 with MC=200 
   % 1.0000    0.9900    1.0000    1.0000    1.0000    0.6500    0.7850    0.8050    0.7750    0.9050
   % 1.0000    1.0000    1.0000  ;  overall accuracy is 91.62 % 
    

   
   
   