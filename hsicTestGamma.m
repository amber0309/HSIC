%This function implements the HSIC test using a Gamma approximation
%to the test threshold

%Arthur Gretton
%03/06/07

%Inputs: 
%        X contains dx columns, m rows. Each row is an i.i.d sample
%        Y contains dy columns, m rows. Each row is an i.i.d sample
%        alpha is the level of the test
%        params.sigx is kernel size for x (set to median distance if -1)
%        params.sigy is kernel size for y (set to median distance if -1)

%Outputs: 
%        thresh: test threshold for level alpha test
%        testStat: test statistic

%Set kernel size to median distance between points, if no kernel specified

%11/01/08 Used new expression for beta independent of m, and
%         m*HSICb as test statistic

function [thresh,testStat,params] = hsicTestGamma(X,Y,alpha,params);

    
m=size(X,1);



%Set kernel size to median distance between points, if no kernel specified.
%Use at most 100 points (since median is only a heuristic, and 100 points
%is sufficient for a robust estimate).
if params.sigx == -1
    size1=size(X,1);
    if size1>100
      Xmed = X(1:100,:);
      size1 = 100;
    else
      Xmed = X;
    end
    G = sum((Xmed.*Xmed),2);
    Q = repmat(G,1,size1);
    R = repmat(G',size1,1);
    dists = Q + R - 2*Xmed*Xmed';
    dists = dists-tril(dists);
    dists=reshape(dists,size1^2,1);
    params.sigx = sqrt(0.5*median(dists(dists>0)));  %rbf_dot has factor of two in kernel
end

if params.sigy == -1
    size1=size(Y,1);
    if size1>100
      Ymed = Y(1:100,:);
      size1 = 100;
    else
      Ymed = Y;
    end    
    G = sum((Ymed.*Ymed),2);
    Q = repmat(G,1,size1);
    R = repmat(G',size1,1);
    dists = Q + R - 2*Ymed*Ymed';
    dists = dists-tril(dists);
    dists=reshape(dists,size1^2,1);
    params.sigy = sqrt(0.5*median(dists(dists>0)));
end



bone = ones(m,1);
H = eye(m)-1/m*ones(m,m);


K = rbf_dot(X,X,params.sigx);
L = rbf_dot(Y,Y,params.sigy);

Kc = H*K*H; %Note: these are slightly biased estimates of centred Gram matrices
Lc = H*L*H;
  
%NOTE: we fit Gamma to testStat*m
testStat = 1/m * sum(sum(Kc'.*Lc));    %%%% TEST STATISTIC: m*HSICb (under H1)
  
varHSIC = (1/6 * Kc.*Lc).^2;
  
varHSIC = 1/m/(m-1)* (  sum(sum(varHSIC)) - sum(diag(varHSIC))  ); 
%second subtracted term is bias correction
  
varHSIC = 72*(m-4)*(m-5)/m/(m-1)/(m-2)/(m-3)  *  varHSIC;  %variance under H0


K = K-diag(diag(K));
L = L-diag(diag(L));

muX = 1/m/(m-1)*bone'*(K*bone);
muY = 1/m/(m-1)*bone'*(L*bone);
  
mHSIC  = 1/m * ( 1 +muX*muY  - muX - muY )         ;  %mean under H0


al = mHSIC^2 / varHSIC;
bet = varHSIC*m / mHSIC;   %NOTE: threshold for hsicArr*m

thresh = icdf('gam',1-alpha,al,bet);    %%%% TEST THRESHOLD
