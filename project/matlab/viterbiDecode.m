function [ yOpt ] = viterbiDecode( p0,pT )
%viterbiDecode implements Viterbi decoding algorithm for Markov chains

%   p0 : initial distribution
%   pT : transition probability

nNodes = size(pT,3)+1;
nStates=length(p0);      % number of hidden states
M=zeros(nStates,nNodes); % table for probabilities
B=zeros(nStates,nNodes); % table for linking nodes(where current node comes from)

M(:,1)=p0;  % prob. of first node equals pi_0
 
for k=2:nNodes
    for kk=1:nStates
        P=reshape(pT(:,:,k-1),nStates,nStates);
        tmp=M(:,k-1).*P(:,kk);  
        [M(kk,k),B(kk,k)]=max(tmp); 
    end
end

% backtracking to find optimal sequence
yOpt=zeros(1,nNodes);   % optimal Viterbi sequence
[~,yOpt(nNodes)]=max(M(:,nNodes));
for k=nNodes-1:-1:1
    yOpt(k)=B(yOpt(k+1),k+1);
end


end
