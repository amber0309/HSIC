%Log likelihood test

%Arthur Gretton
%16/02/08

%Equiprobable space partitioning code (c) Kenji Fukumizu  




%Inputs: 
%        X contains dx columns, L rows. Each row is an i.i.d sample
%        Y contains dy columns, L rows. Each row is an i.i.d sample
%        params.q is number of partitions per dimension (Ku and Fine use 4),


%Outputs: 
%        thresh: test threshold for level alpha test
%        testStat: test statistic

function [thresh,testStat] = likeRatioTest(X,Y,alpha,params);

  

  
q=params.q;

L = size(X,1);
dx = size(X,2);
dy = size(Y,2);


[idxX,blockX] = EqualPartition(X,q);
[idxY,blockY] = EqualPartition(Y,q);

blockX = [cumsum(blockX)-blockX(1) sum(blockX)];
blockY = [cumsum(blockY)-blockY(1) sum(blockY)];

testStat = 0;

for indCellX = 1:q^dx
  for indCellY = 1:q^dy

    %Find indices of X and Y that fall in the current cell
    indSet = intersect(idxX(blockX(indCellX)+1:blockX(indCellX+1)),idxY(blockY(indCellY)+1:blockY(indCellY+1)));
    Nj = length(indSet);  %Note: Nj/L is the empirical meaure of
                          %the weight of cell A*B.
    %q^dx is weight of an individual marginal cell, given
    %equiprobable partitioning
    
    
    if Nj~=0
      %Avoid error with log zero if no samples in a bin. 
      %Note that in teststat
      %term, this is multiplied by zero and vanishes.
       testStat = testStat + 2* (Nj/L) * log ((Nj/L)/(1/q^dx/q^dy));
    end
    
  end
    
end


%DEGREES OF FREEDOM: the next line assumes dx=dy
%Thus, both X and Y are partitioned into q^dx bins



thresh =  ( sqrt(2*q^dx*q^dy)*icdf('normal',1-alpha,0,1) + q^dx*q^dy )/L;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% EqualPartition()
% Recursive partitioninig of the dimension with the equal number of data.
%
% Date: June 1, 2007
% Version: 0.9
% Author: Kenji Fukumizu
% Affiliation: Institute of Statistical Mathematics, ROIS
% (c) Kenji Fukumizu
%
% Input:
% X - data matrix (row=data, column=dimension)
% K - Number of partitions for each dimension
% (Total number of bins = K^{dim X})
% Output:
% idx - sorted index of X in the order of the bins
% block - the number of data in the bins
%
% X(idx(block(j-1)+1:block(j)),:) are the data in j-th bin (for j>=2)
%

function [idx block] = EqualPartition(X,K)
%function [cellCount,partitionInd]=getCellCount(sample,q)

[N dim]=size(X);
idx=1:N;
prevblock=[N];
for h=1:dim
    numblk=length(prevblock);
    start=1;
    b_end=cumsum(prevblock,2);
    tmpblock=[];
    
    for i=1:numblk
        subidx=idx(start:b_end(i));
        [v is]=sort(X(subidx,h));
        nk=length(subidx);
        num=floor(nk/K);
        if num<=1
            error('Use smaller number of partitions (K): EqualPartition()');
            % Each bin must contain at least two data.
        end
        subblk=num*ones(1,K)+[zeros(1,K-mod(nk,K)) ones(1,mod(nk,K))];
        tmpblock=[tmpblock subblk];
        idx(start:b_end(i))=subidx(is);
        start=b_end(i)+1;
    end
    prevblock=tmpblock;
end
block=tmpblock;





