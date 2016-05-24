%Log likelihood test

%Equiprobable space partitioning (c) Kenji Fukumizu  

%Arthur Gretton
%16/02/08



%Inputs: 
%        X contains dx columns, L rows. Each row is an i.i.d sample
%        Y contains dy columns, L rows. Each row is an i.i.d sample
%        params.q is number of partitions per dimension (Ku and Fine use 4),
%        params.bootForce=1: run bootstrap. If 0: only bootstrap if no
%        previous threshold file found
%        params.shuff: number of shuffles used in bootstrap


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
%testStatPears = 0;    %CHECK: comparison of Pearson and log
                       %likelihood statistics

for indCellX = 1:q^dx
  for indCellY = 1:q^dy
    indSet = intersect(idxX(blockX(indCellX)+1:blockX(indCellX+1)),idxY(blockY(indCellY)+1:blockY(indCellY+1)));
    Nj = length(indSet);     
    
    if Nj~=0
      %Avoid error with log zero if no samples in a bin. 
      %Note that in teststat
      %term, this is multiplied by zero and vanishes.
      testStat = testStat + 2* (Nj/L) * log ((Nj/L)/(1/q^dx/q^dy));
    end
  end
    
end


thresh = getBootThresh(idxX,blockX,idxY,blockY,dx,dy,alpha,params);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [thresh] = getBootThresh(idxX,blockX,idxY,blockY,dx,dy,alpha,params)

%File stored by number of points, number of dimensions, **number of subdivisions in space**




L=length(idxX);
q=params.q;
threshFileName = strcat('likeRatioThresh',num2str(L),'_',num2str(dx),'_',num2str(q));


if ~exist(strcat(threshFileName,'.mat'),'file') || params.bootForce==1
  
  
  
%  disp(strcat('Generating new threshold: ',threshFileName))
  
  likeRatioDistArr=zeros(params.shuff,1);
  
  for whichSh=1:params.shuff

    %Re-order indices of Y variables to assign them to random blocks
    idxY = idxY(randperm(L));
    testStat = 0;
    
    for indCellX = 1:q^dx
      for indCellY = 1:q^dy   

	indSet = intersect(idxX(blockX(indCellX)+1:blockX(indCellX+1)),idxY(blockY(indCellY)+1:blockY(indCellY+1)));
	Nj = length(indSet);
	if Nj~=0 %avoid log zero
	  testStat = testStat +  2* (Nj/L) * log ((Nj/L)/(1/q^dx/q^dy));
        end
	
      end
    end
    
    likeRatioDistArr(whichSh) = testStat;
    

    
  end 
  
  likeRatioDistArr = sort(likeRatioDistArr);
  thresh = likeRatioDistArr(round((1-alpha)*params.shuff));
  save(threshFileName,'thresh','likeRatioDistArr');  
  
  
else
  load(threshFileName);
end



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





