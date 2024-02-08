close all
clear all
elev = load( '../resources/maps/mapaBig1.dat' );

[X,Y] = meshgrid(-6.0:0.02:5.99,-6.0:0.02:5.99);

surf(X,Y,elev);
xlabel('x [m]')
ylabel('y [m]')
zlabel('z [m]')
%  colormap('autumn');
colormap ("default");
shading interp;
%  axis equal;
view(159,38)

%  print -deps -color map.eps
%  print -dpng -color map.png