% different Q and M wrt performances
% record by changing Q, and M individually in the main function
% For each Q=1,2,..., we record accuracy with M from 1:10.

close all;
clear;clc

Record=[59.74 76.15 91.28 85.13 83.85 91.54 90.51 88.46 86.92 87.44;
        79.49 85.38 94.36 89.23 89.49 94.87 90.51 84.36 86.41 90.26;
        88.72 87.44 88.97 92.56 91.10 91.03 84.87 86.67 90.51 86.67;
        87.44 91.79 89.49 91.28 83.85 90.00 84.87 82.05 83.08 84.87;
        91.79 84.87 87.69 87.69 85.13 86.15 84.87 89.23 84.10 81.54;
        91.03 92.31 85.13 89.23 90.26 80.00 84.10 84.10 NaN   NaN  ;
        90.51 87.44 85.90 84.10 74.10 87.44 81.54 NaN   NaN   NaN  ;
        89.23 81.54 76.92 87.69 83.59 NaN   NaN   NaN   NaN   NaN  ;   
       ];
   
 set(0,'DefaultTextFontName','Times',...
'DefaultTextFontSize',16,...
'DefaultAxesFontName','Times',...
'DefaultAxesFontSize',16,...
'DefaultLineLineWidth',2,...
'DefaultLineMarkerSize',12);
   
   M=1:10;
   plot(M,Record(1,:),'b.');  hold on;
   plot(M,Record(2,:),'g-o'); hold on;
   plot(M,Record(3,:),'r.-'); hold on;
   plot(M,Record(4,:),'m-');  hold on;
   plot(M,Record(5,:),'y-o'); hold on;
   plot(M,Record(6,:),'k.');  hold on;
   plot(M,Record(7,:),'c<');  hold off;
   %plot(M,Record(8,:),'b-s'); hold off;
   
   xlabel('number of joint states ');
   ylabel('recognition accuracy (%)');
    axis([1 10 78 95]);
   legend('Q=1,M=1:10','Q=2,M=1:10','Q=3,M=1:10','Q=4,M=1:10','Q=5,M=1:10', 'Q=6,M=1:10','Q=7,M=1:10');% ,'Q=8,M=1:10
  
   