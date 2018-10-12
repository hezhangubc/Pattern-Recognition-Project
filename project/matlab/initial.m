function [ A,B,Cs,Cm ] = initial( Data, N, Ks, Km)
%INITIAL initialization pre-processing for GMM-HMM para. training
%   [ A,B ] = initial( Data, N, Ks, Km)
%   Data: N skeleton joints' position of one person
%   N :   number of skeleton joints
%   Ks:   number of pose states
%   Km:   number of joint states
%   A:    prob. transition matrix
%   B:    obser. prob. tran. matr.
%   mu:   mean of each state pair
%   sigma: variance of each state pair

% initialization step
[T,~]=size(Data);
A=zeros(Ks,Ks);     
B=zeros(Ks,Km,N);
Cm=zeros(N,T);  % class label matrix, dim: joints by  time sampling length

% Step 1: K-means for clustering hidden joints one by one
for k=1:N
    [IDX,~]=kmeans(Data(:,(k-1)*3+1:k*3),Km);
    Cm(k,:)=IDX';
end

% Step 2: K-means for clustering hidden poses one by one
[IDX,~]=kmeans(Cm',Ks);  % joint states as features, convert to row      ## Cm here might be replaced by positions ##
Cs=IDX';                 % class label vector for pose states

% find A, B using MLE
% 1) find the prob. transition matrix A 3*3
for k=2:T
    A(Cs(k-1),Cs(k))=A(Cs(k-1),Cs(k))+1;
end
% normalization
for k=1:Ks
    A(k,:)=A(k,:)/sum(A(k,:));
end

% 2) find the obser. prob. transition matrix B 3*3*11
for j=1:N      % joint #
    for k=1:T  % time index
        B(Cs(k),Cm(j,k),j)=B(Cs(k),Cm(j,k),j)+1;
%         mu(:,Cs(k),Cm(j,k),j)=mu(:,Cs(k),Cm(j,k),j)+
    end
    % normalization
    for kk=1:Km
        B(kk,:,j)=B(kk,:,j)/sum(B(kk,:,j));
    end
end


end

