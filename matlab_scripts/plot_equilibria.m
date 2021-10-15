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
my_size = (3+4)*N*N;
precision = 'double';
skip = 0;
data = fread(fileID, my_size, precision, skip);

state = data(1:3*N*N);
F = data(3*N*N + (1:4*N*N));



[state, F] = rescale_state(state, F, N);



my_size = max_iterations;
precision = 'double';
skip = 0;
gmres_res = fread(fileID, my_size, precision, skip);
normF  = fread(fileID, my_size, precision, skip);
normJtF = fread(fileID, my_size, precision, skip);
fclose(fileID);






tiledlayout(2,3);

nexttile
%First figure should be convergence information
loglog( 1:max_iterations, normF, 'o' );
hold on
loglog( 1:max_iterations, normJtF, 'o' );
loglog( 1:max_iterations, gmres_res, 'o' );
hold off
title("Newton-GMRES error")
legend("|F|", "|J'*F|", "GMRES residual")
ylabel("magnitude");
xlabel("iteration");
pause(1e-1);



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
imagesc( log(abs(fft_w)) );
title('Fourier Spectra of vorticity')
pbaspect([1 1 1]);
colorbar();
adjust_caxis( fft_w )


nexttile
fft_w = fftshift( fft2(u) );
imagesc( log(abs(fft_w)) );
title('Fourier Spectra of u')
pbaspect([1 1 1]);
colorbar();
adjust_caxis( fft_w )

nexttile
fft_w = fftshift( fft2(v) );
imagesc( log(abs(fft_w)) );
title('Fourier Spectra of v')
pbaspect([1 1 1]);
colorbar();
adjust_caxis( fft_w )


norm(F)/norm(state)



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
  caxis( [min(log_data(good)) max(log_data(good))] );
end