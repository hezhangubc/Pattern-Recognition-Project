% K-means method trial

close all;
clear;clc

N=1e3;

Mu=[-2 2;0 0]; 
Sigma=.5*eye(2);

X=zeros(N,2);  % 2 by N, x_k rowwise

X(1:500,:)=mvnrnd(Mu(:,1),Sigma,500);
X(501:1000,:)=mvnrnd(Mu(:,2),Sigma,500);

scatter(X(:,1),X(:,2))

Mu_est=[-2 2;1 1];   % initialization
Rnk=zeros(N,2);
Rnk(1:200,1)=ones(200,1); 
Rnk(201:1000,2)=ones(800,1); 

Iter_max=100;
Mu_est_old=Mu_est;
epsilon=1e-4;

for iter=1:Iter_max
    for n=1:N
        Rnk(n,:)=zeros(1,2);
        [~,I]=min([norm(X(n,:)-Mu_est(:,1)'),norm(X(n,:)-Mu_est(:,2)')]);
        Rnk(n,I)=1;
    end
    
    Mu_est=X'*Rnk;
    Mu_est(:,1)=Mu_est(:,1)/sum(Rnk(:,1));
    Mu_est(:,2)=Mu_est(:,2)/sum(Rnk(:,2));
    if norm(Mu_est-Mu_est_old)/norm(Mu_est_old)<epsilon, break;
    else, Mu_est_old=Mu_est;
    end
end

iter
Mu_est
