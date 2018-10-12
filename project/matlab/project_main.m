% Matlab code for EECE 562 project

close all;
clear;clc

%% parameter setup
N=11;               % number of skeleton joints
Ks=3;               % number of pose states 
Km=3;               % number of middle(joint) states 
T=500;              % time slice recorded
Mu=zeros(3,Ks,Km,N);
Sigma=zeros(3,3,Ks,Km,N);
% A=zeros(Ks,Ks);     % prob. transition matrix of pose states
% B=zeros(Ks,Km,N);   % obser. prob. transition matrix 

%% EM to train GMM based HMM
%  ##  data from person 1 ##

% initialization 
load person_1.mat
Data=still;            % take 'still' activity for example, modify it for other activities
% Data=Data(1:end-1,:);  % delete last row (END symbol, NaN) 
Data=Data(1:T,:);

[A,B,Cs,Cm]=initial(Data, N, Ks, Km);   % do initialization pre-processing 

% EM
for j=1:1 
% 1) find mu for each state pair
    mu=zeros(3,Ks,Km); % 3: position dimension
    sigma=zeros(3,3,Ks,Km);
    count=zeros(Ks,Km);
    for t=1:size(Data,1)    % find mu
        mu(:,Cs(t),Cm(j,t))= mu(:,Cs(t),Cm(j,t))+Data(t,(j-1)*3+1:j*3)';
        count(Cs(t),Cm(j,t))=count(Cs(t),Cm(j,t))+1;
    end
 % average  the mu 
    for k1=1:Ks
        for k2=1:Km
            if count(k1,k2)==0, count(k1,k2)=1; end   % in case of zero denom.
            mu(:,k1,k2)= mu(:,k1,k2)/count(k1,k2);
        end
    end
 % 2) find sigma for each state pair       
    for t=1:size(Data,1)  % find sigma
         sigma(:,:,Cs(t),Cm(j,t))=sigma(:,:,Cs(t),Cm(j,t))+(Data(t,(j-1)*3+1:j*3)'-mu(:,Cs(t),Cm(j,t)))*(Data(t,(j-1)*3+1:j*3)'-mu(:,Cs(t),Cm(j,t)))';
    end
% average the sigma
    for k1=1:Ks
        for k2=1:Km
            if sigma(:,:,k1,k2)==0, sigma(:,:,k1,k2)=eye(3); end
            sigma(:,:,k1,k2)= sigma(:,:,k1,k2)/count(k1,k2);
        end
    end

%     [A,tmp]=ChmmGmm(Data(:,(j-1)*3+1:j*3)',Ks,Km,ones(3,1)/Ks,);
    tmp=reshape(B(:,:,j),Ks,Km);
    [LL, prior, A, mu, sigma, tmp] = ...
    mhmm_em(Data(:,(j-1)*3+1:j*3)', ones(3,1)/Ks, A, mu, sigma, tmp, 'max_iter', 1);
    % update para.
%     B(:,:,j)=tmp;
%     Mu(:,:,j)=mu;
%     Sigma(:,:,:,:,j)=sigma;
end

%  ##  data from person 2 ##
% load person2_mat
% Data=still;            
% Data=Data(1:end-1,:);  
% 
% % EM
% for j=1:N 
%     mu=reshape(Mu(:,:,j),3,Ks,Km);
%     sigma=reshape(Sigma(:,:,:,:,j),3,3,Ks,Km);
%     tmp=reshape(B(:,:,j),Ks,Km);
%     [LL, prior, A, mu, sigma, tmp] = ...
%     mhmm_em(Data(:,(j-1)*3+1:j*3)', ones(3,1)/Ks, A, mu, sigma, tmp, 'max_iter', 5);
%     % update para.
%     B(:,:,j)=tmp;
%     Mu(:,:,j)=mu;
%     Sigma(:,:,:,:,j)=sigma;
% end

% %  ##  data from person 3 ##
% load person3_mat
% Data=still;            
% Data=Data(1:end-1,:); 
% 
% % EM
% for j=1:N 
%     [A,B]=chmmgmm();
% end
% 
% %  ##  data from person 4 ##
% load person4_mat
% Data=still;            
% Data=Data(1:end-1,:); 
% 
% % EM
% for j=1:N 
%     [A,B]=chmmgmm();
% end

