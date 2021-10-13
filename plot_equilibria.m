clf


fileID = fopen('solution.bin');
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
%state = rescale_state(state, N);

my_size = max_iterations;
precision = 'double';
skip = 0;
gmres_res = fread(fileID, my_size, precision, skip);
normF  = fread(fileID, my_size, precision, skip);
normJtF = fread(fileID, my_size, precision, skip);
fclose(fileID);


figure(1);
pause(1e-1);
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




figure(2);
pause(1e-1);

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

function state2 = rescale_state(state, N)
  u = state( 0*N*N + (1:N*N) );
  v = state( 1*N*N + (1:N*N) );
  w = state( 2*N*N + (1:N*N) );
  
  val = max(abs(w));
  state2 = state/val;
end
