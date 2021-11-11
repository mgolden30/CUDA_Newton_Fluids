%{
Script to find compact difference formulas
%}

n     = 4;
order = 4*n-2;

A = zeros(2*n, order/2);
A = sym(A);
for i = 1:n
  for j=1:order/2
    A(     i, j) = 2/factorial(2*j-1)*((2*i-1)/2)^(2*j-1);
    A( n + i, j) = 2/factorial(2*j-2)*((2*i-1)/2)^(2*j-2);
  end
end

%{
A = [ 1,   2/factorial(3) * (1/2)^3,   2/factorial(5) * (1/2)^5;
      3,   2/factorial(3) * (3/2)^3,   2/factorial(5) * (3/2)^5;
      
      2,   2/factorial(2) * (1/2)^2,   2/factorial(4) * (1/2)^4;
      2,   2/factorial(2) * (3/2)^2,   2/factorial(4) * (3/2)^4;
      ];
%}    

[Q,R] = qr(A);
%A = A';
%[U,S,V] = svd(A);
%A*V(:,end)

double(Q(:,end))

double(A')*double(Q(:,end))

weights = double(Q(:,end));

str = "#define Qx(u,i,j)     (";
for i=1:n
  str = str + sprintf("%.50f * (u[IDXP(i+%d,j)] + u[IDXP(i+%d, j)])", -weights(i+n), i, 1-i );
  if i~= n
    str = str + " + "; 
  end
end
str = str + ")\n";

str = str + "#define Qy(u,i,j)     (";
for i=1:n
  str = str + sprintf("%.50f * (u[IDXP(i,j+%d)] + u[IDXP(i, j+%d)])", -weights(i+n), i, 1-i );
  if i~= n
    str = str + " + "; 
  end
end
str = str + ")\n";

str = str + "#define Px(u,i,j)     (";
for i=1:n
  str = str + sprintf("%.50f * (u[IDXP(i+%d,j)] - u[IDXP(i+%d, j)])", weights(i), i, 1-i );
  if i~= n
    str = str + " + "; 
  end
end
str = str + ")/dx\n";

str = str + "#define Py(u,i,j)     (";
for i=1:n
  str = str + sprintf("%.50f * (u[IDXP(i,j+%d)] - u[IDXP(i, j+%d)])", weights(i), i, 1-i );
  if i~= n
    str = str + " + "; 
  end
end
str = str + ")/dx\n";