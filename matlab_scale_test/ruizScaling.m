function [D] = ruizScaling(M,scaleNorm,maxIter,scalingFlags)

%Apply the Ruiz method to a matrix M
%
%   Usage : [D] = ruizScaling(M,scaleNorm,maxIter,scalingFlags)
%
%   Inputs : M - a symmetric matrix
%            scaleNorm    - the norm to use in the scaling (1,2,inf). 
%                           Default is inf.
%            maxIter      - maximum iterations. Default is 15.
%            scalingFlags - logical vector indicating whether each column
%                           should have scaling applied to it.
%
%   Outputs: D is diagonal scaling matrix so that D*M*D is nice.

SCALING_REG = 1e-06;

if(nargin < 2 | isempty(scaleNorm)), scaleNorm      = inf; end
if(nargin < 3 | isempty(maxIter)),   maxIter        = 15; end
if(nargin < 4 | isempty(maxIter)),   scalingFlags   = ones(size(M,1)); end

%size of the matrix
[m,n] = size(M); assert(m==n);

%initialize scaling
d      = ones(n,1);
d_temp = ones(n,1);

for i = 1:maxIter
    for j = 1:n
        norm_col_j = norm(M(:,j),scaleNorm);
        
        if(norm_col_j > SCALING_REG && scalingFlags(j))
            d_temp(j) = 1./sqrt(norm_col_j);
        end
    end
    S_temp = spdiags(d_temp,0,n,n);
    d      = d.*d_temp;
    M      = S_temp*M*S_temp;
end

D = spdiags(d,0,n,n);