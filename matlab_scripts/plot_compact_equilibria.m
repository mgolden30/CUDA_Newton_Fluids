clf


fileID = fopen('../solution.bin');
%fileID = fopen('euler1.bin');


%First read sidelength
my_size = 1;
precision = 'int';
skip = 0;
N              = fread(fileID, my_size, precision, skip);
max_iterations = fread(fileID, my_size, precision, skip);



%Now with this info, read the rest of the state
my_size = (9)*N*N;
precision = 'double';
skip = 0;
state = fread(fileID, my_size, precision, skip);
F     = fread(fileID, my_size, precision, skip);

%[state, F] = rescale_state(state, F, N);



my_size = max_iterations;
precision = 'double';
skip = 0;
gmres_res = fread(fileID, my_size, precision, skip);
normstate = fread(fileID, my_size, precision, skip);
normF     = fread(fileID, my_size, precision, skip);
%normJtF   = fread(fileID, my_size, precision, skip);
fclose(fileID);


figure(1)
tiledlayout(2,3);

nexttile
%First figure should be convergence information
loglog( 1:max_iterations, normF, 'o' );
hold on
loglog( 1:max_iterations, normstate, 'o' );
%loglog( 1:max_iterations, normJtF, 'o' );
loglog( 1:max_iterations, gmres_res, 'o' );
loglog( 1:max_iterations, normF./normstate, 'o' );
hold off
title("Newton-GMRES error")
legend({"|F|", "|state|", "GMRES residual", "|F|/|state|"}, ...
    'Location', 'southwest' )
ylabel("magnitude");
xlabel("iteration");
pause(1e-1);
ylim([1e-4, 1e2 ]);


nexttile
%Now plot the solution

dx = 2*pi/N;
my_ticks = (0:N-1)/N*2*pi;
[x,y] = meshgrid( my_ticks, my_ticks );

u = state( 0*N*N + (1:N*N) );
v = state( 1*N*N + (1:N*N) );
w = state( 2*N*N + (1:N*N) );

u = reshape( u, [N,N] );
v = reshape( v, [N,N] );
w = reshape( w, [N,N] );

imagesc( my_ticks, my_ticks, w );
pbaspect([1 1 1]);
hold on
  d = round(N/32);
  quiver(x(1:d:end, 1:d:end), y(1:d:end, 1:d:end), u(1:d:end, 1:d:end), v(1:d:end, 1:d:end), 'Color', 'black' );
hold off
wmax = max(max(abs(w)));
colorbar( 'Ticks', wmax*(-1:1) );
caxis(wmax*[-1 1]);
colormap bluewhitered
set(gca, 'YDir','normal')
xticks([0, 2*pi-dx]);
xticklabels({'0','2\pi'})
yticks([0, 2*pi-dx]);
yticklabels({'0','2\pi'})
xlabel('x')
ylabel('y')
hYLabel = get(gca,'YLabel');
set(hYLabel,'rotation',0,'VerticalAlignment','middle')
%title('equilibrium of Navier-Stokes, F_\omega = 4cos(4y) and \nu=1/60')
title('equilibrium of the Euler equations')

nexttile
plot(F);
title('F');


nexttile
fft_w = fftshift( fft2(w) );
imagesc( log10(abs(fft_w)) );
title('Fourier Spectra of vorticity')
pbaspect([1 1 1]);
colorbar();
adjust_caxis( fft_w )


nexttile
fft_w = fftshift( fft2(u) );
imagesc( log10(abs(fft_w)) );
title('Fourier Spectra of u')
pbaspect([1 1 1]);
colorbar();
adjust_caxis( fft_w )

nexttile
fft_w = fftshift( fft2(v) );
imagesc( log10(abs(fft_w)) );
title('Fourier Spectra of v')
pbaspect([1 1 1]);
colorbar();
adjust_caxis( fft_w )


norm(F)/norm(state)

check_fft(state, N);
check_residual( F, N );
function [state2, F2] = rescale_state(state, F, N)
  u = state( 0*N*N + (1:N*N) );
  v = state( 1*N*N + (1:N*N) );
  w = state( 2*N*N + (1:N*N) );
  
  Ff1 = F( 0*N*N + (1:N*N) );
  Ff2 = F( 1*N*N + (1:N*N) );
  Ff3 = F( 2*N*N + (1:N*N) );
  Ff4 = F( 3*N*N + (1:N*N) );
  
  val = max(abs(w));
  state2 = state/val;
  F2 = [Ff1/val; Ff2/val; Ff3/val; Ff4/val/val];
end

function adjust_caxis( data )
  log_data = log(abs(data));
  good = log_data > -10;
  caxis( [ 1.5*min(log_data(good)) max(log_data(good))] );
end

function check_residual( state, N )
  figure(3);
  
  tiledlayout(3,3);
  
  for i=1:9
    nexttile
    u = reshape( state( (i-1)*N*N + (1:N*N) ), [N,N] );       
    imagesc( u );
    title( sprintf('residual %d', i) );
    pbaspect([1 1 1]);
    colorbar();
    
    pause(1e-1)
  end
end


function check_fft( state, N )
  figure(2);
  
  tiledlayout(3,3);
  
  for i=1:9
    nexttile
    u = reshape( state( (i-1)*N*N + (1:N*N) ), [N,N] ); 
      
    fft_w = fftshift( fft2(u) );
    imagesc( log10(abs(fft_w)) );
    title( sprintf('Fourier Spectra of field number %d', i) );
    pbaspect([1 1 1]);
    colorbar();
    adjust_caxis( fft_w );
    
    pause(1e-1)
  end
end