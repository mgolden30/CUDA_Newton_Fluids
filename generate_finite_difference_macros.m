%Sorry that this script is a mess. It does its job.

n = 6; %order of derivatives
str = stencil_string(n); %string that contains C macros
return

%Don't read further than this if you don't care about details.
















N = 64;
dx = 2*pi/N;

error = [];
nrange = 2:2:30;

for n=nrange
  %w = centered_difference_stencil2(n);
  w = interp_coeff(n);
  x = 0.5;
  %temp = (cos(x+(1:n/2)*dx) - cos(x-(1:n/2)*dx))/(2*dx);
  %temp = (cos(x+(1:2:n)/2*dx) - cos(x-(1:2:n)/2*dx))/(dx);
  %error = [ error abs(temp*w + sin(x)) ];
  
  %For testing interpolation
  temp = (cos(x+(1:2:n)/2*dx) + cos(x-(1:2:n)/2*dx))/2;
  error = [error abs(temp*w - cos(x))];
end

semilogy(nrange, error, 'o');
yline(1e-16);
ylim([1e-18, 1e-2])



function w = centered_difference_stencil(n)
  %{
  Computes the rational coefficients for a centered difference of order n
  %}

  assert( mod(n,2) == 0 ); %Only even n
  A = (1:n/2).^((1:2:n)');
  b = zeros(n/2,1);
  A = sym(A);
  b(1) = 1;
  w = linsolve(A,b);
end


function w = centered_difference_stencil2(n)
  %{
  Computes the rational coefficients for a centered difference of order n,
  but for a grid point. 
  %}

  assert( mod(n,2) == 0 ); %Only even n
  A = ((1:2:n)).^((1:2:n)');
  b = zeros(n/2,1);
  A = sym(A);
  b(1) = 1;
  w = linsolve(A,b);
end

function w = interp_coeff(n)
  %{
  Computes the rational coefficients for an interpolation of order n
  %}

  assert( mod(n,2) == 0 ); %Only even n
  A = ((1:2:n)).^((0:2:n-1)');
  b = zeros(n/2,1);
  A = sym(A);
  b(1) = 1;
  w = linsolve(A,b);
end


function i = good_denom(w)
  for i=1:10000
    if floor(i*w) == i*w
      return
    end
  end
end

function str = uu(i,j)
  str = "u[IDXP("+i+","+j+")]";
end

function str = stencil_string(n)
  w = centered_difference_stencil(n);
  denom = good_denom(w);
  w2 = denom*w; %Integer coefficients
  
  %{
  First Let's construct macros for Dxa and Dya
  %}
  
  str = "#define Dxa(u,i,j)    ( ";
  for i=1:size(w)
    str = str + double(w2(i)) + "*(u[IDXP(i+" + i + ",j)]" + " - u[IDXP(i-" + i + ",j)])";
    if i ~= size(w)
      str = str + " + "; 
    end
  end
  str = str + ")/(" + 2*denom + "*dx)\n";
  
  str = str + "#define Dya(u,i,j)    ( ";
  for i=1:size(w)
    str = str + double(w2(i)) + "*(u[IDXP(i,j+" + i + ")]" + " - u[IDXP(i,j-" + i + ")])";
    if i ~= size(w)
      str = str + " + "; 
    end
  end
  str = str + ")/(" + 2*denom + "*dx)\n";
  
  str = str + "#define Dxa_prod(u,v,i,j)    ( ";
  for i=1:size(w)
    str = str + double(w2(i)) + "*( u[IDXP(i+" + i + ",j)]*v[IDXP(i+" + i + ",j)]" ...
                              + " - u[IDXP(i-" + i + ",j)]*v[IDXP(i-" + i + ",j)])";
    if i ~= size(w)
      str = str + " + "; 
    end
  end
  str = str + ")/(" + 2*denom + "*dx)\n";
  
  
  str = str + "#define Dya_prod(u,v,i,j)    ( ";
  for i=1:size(w)
    str = str + double(w2(i)) + "*( u[IDXP(i,j+" + i + ")]*v[IDXP(i,j+" + i + ")]" ...
                              + " - u[IDXP(i,j-" + i + ")]*v[IDXP(i,j-" + i + ")])";
    if i ~= size(w)
      str = str + " + "; 
    end
  end
  str = str + ")/(" + 2*denom + "*dx)\n";
  
  %{
  Now macros for Dxb and Dyb
  %}
  w1 = centered_difference_stencil2(n);
  w2 = interp_coeff(n);
  d1 = good_denom(w1);
  d2 = good_denom(w2);
  w1 = w1*d1;
  w2 = w2*d2;
  
  str = str + "#define Dxb(u,i,j)    ( ";
  for i=1:size(w1)
    for j=1:size(w2)
      %n = 4 so size(w1) = 2
      %i   =   1  2
      %idx =   1  2 %i
      %opp =   0 -1 %1-i
      
      str = str + double(w1(i)*w2(j)) + "*(" + uu("i+"+i,     "j+"+j    ) ...
                                     + " - " + uu("i+"+(1-i), "j+"+j    ) ...
                                     + " + " + uu("i+"+i,     "j+"+(1-j)) ...
                                     + " - " + uu("i+"+(1-i), "j+"+(1-j)) + ")";
     if i~=size(w1) | j~=size(w2) 
       str = str + "+";
     end
    end
  end
  str = str + ")/(" + 2*d1*d2 + "*dx)\n";
  
  str = str + "#define Dyb(u,i,j)    ( ";
  for i=1:size(w2)
    for j=1:size(w1)
      %n = 4 so size(w1) = 2
      %i   =   1  2
      %idx =   1  2 %i
      %opp =   0 -1 %1-i
      
      str = str + double(w1(j)*w2(i)) + "*(" + uu("i+"+i,     "j+"+j    ) ...
                                     + " + " + uu("i+"+(1-i), "j+"+j    ) ...
                                     + " - " + uu("i+"+i,     "j+"+(1-j)) ...
                                     + " - " + uu("i+"+(1-i), "j+"+(1-j)) + ")";
     if i~=size(w2) | j~=size(w1) 
       str = str + "+";
     end
    end
  end
  str = str + ")/(" + 2*d1*d2 + "*dx)\n";

  fprintf(str);
end