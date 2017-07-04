function A = cscToSparse(D)

%Convert CSC format data from OSQP workspace back to a sparse matrix

%Field names are
%nzmax, m, n, p, i, x, nz

if(~any(D.p))
    %empty matrix
    A = sparse(D.m,D.n);
    return;
end

[~,ist,jst,ast] = cc_to_st(D.m, D.n, length(D.x), D.i, D.p, D.x);
A = sparse(ist+1,jst+1,ast,D.m,D.n);


end



function [ nst, ist, jst, ast ] = cc_to_st ( m, n, ncc, icc, ccc, acc )

%*****************************************************************************80
%
%% CC_TO_ST converts sparse matrix information from CC to ST format.
%
%  Discussion:
%
%    Only JST actually needs to be computed.  The other three output 
%    quantities are simply copies.  
%    
%  Licensing:
%
%    This code is distributed under the GNU LGPL license. 
%
%  Modified:
%
%    23 July 2014
%
%  Author:
%
%    John Burkardt
%
%  Parameters:
%
%    Input, integer M, the number of rows.
%
%    Input, integer N, the number of columns.
%
%    Input, integer NCC, the number of CC elements.
%
%    Input, integer ICC(NCC), the CC rows.
%
%    Input, integer CCC(N+1), the CC compressed columns.
%
%    Input, real ACC(NCC), the CC values.
%
%    Output, integer NST, the number of ST elements.
%
%    Output, integer IST(NST), JST(NST), the ST rows and columns.
%
%    Output, real AST(NST), the ST values.
%
  nst = 0;

  if ( ccc(1) == 0 )

    jlo = 0;
    jhi = n - 1;
  
    for j = jlo : jhi

      klo = ccc(j+1);
      khi = ccc(j+2) - 1;

      for k = klo : khi

        nst = nst + 1;
        ist(nst) = icc(k+1);
        jst(nst) = j;
        ast(nst) = acc(k+1);

      end

    end

  else

    jlo = 1;
    jhi = n;
  
    for j = jlo : jhi

      klo = ccc(j);
      khi = ccc(j+1) - 1;

      for k = klo : khi

        nst = nst + 1;
        ist(nst) = icc(k);
        jst(nst) = j;
        ast(nst) = acc(k);

      end

    end

  end

  return
end