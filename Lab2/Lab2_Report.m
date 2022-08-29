clear all;
close all;
%% Edge Detection
% 3.1 a) 
clear all;
Pc = imread('resource\macritchie.jpg');
whos Pc
P = rgb2gray(Pc);
whos P
figure;colormap('gray');imshow(uint8(P));title("Macritchie gray")

% 3.1 b) 
%Sobel
s_h=[-1 0 1;
     -2 0 2;
     -1 0 1;]
s_v=[-1 -2 -1;
     0 0 0;
     1 2 1;]
%Verticle Filter
Pv=conv2(P,s_v);
%Horizontal Filer
Ph=conv2(P,s_h);

% 3.1 c)
%Magnitude of Gradient Map
P2=(Pv.^2.+Ph.^2).^0.5

figure( 'Position', [10 10 900 600]);
subplot(2,2,1);colormap('gray');imshow(uint8(Ph));title("Macritchie Sobel horizontal filter")
hold on
subplot(2,2,2);colormap('gray');imshow(uint8(Pv));title("Macritchie Sobel verticle filter")
subplot(2,2,3:4);colormap('gray');imshow(uint8(P2));title("Macritchie Sobel edge detector")
hold off

% 3.1 d)Thresholding
min_P2=double(min(P2(:)))%Min intensity=13
max_P2=double(max(P2(:)))%Max intensity=204

%contrast stretching 
P2C = (double(P2(:,:))-min_P2).*(255/(max_P2-min_P2));
%checking min max of P2
min(P2C(:)), max(P2C(:)) %0, 255

t=OTSU_B(P2C,true);
P2t=P2C>t;
[count,bins]=imhist(uint8(P2C),256);

figure( 'Position', [10 10 900 600]);
colormap('gray');imshow(P2t);title("Macritchie Sobel edge detector")
hold off;

t_varry=linspace(10,220,8)
t_varry=[t_varry(1:2),t,t_varry(3:7)]
figure( 'Position', [10 10 1440 960]);
for i = 1 : size(t_varry,2)    
    P2t=P2C>t_varry(i); 
    hold on
    subplot(2,size(t_varry,2)/2,i);colormap('gray');imshow(P2t);
    %A=290/2,B=50;
    %subplot(2,size(t_varry,2)/2,i);colormap('gray');imshow(P2t(A:A+292/4,B:B+360/4));
    if i==3
        title({"Sobel" "t= " t_varry(i),"OTSU"})
    else
        title(["Sobel" "t= " t_varry(i)])
    end
end
hold off

% 3.1 e)
tl=0.04, th=0.1, sigma=1.0;

E = edge(P,'canny',[tl th],sigma); 
figure;colormap('gray');imshow(E);title("Macritchie Sobel edge detector")

% 3.1 e)i) Varry Sigma
sigma_varry=linspace(1,5,5);
figure( 'Position', [10 10 1440 960]);
for i = 1 : size(sigma_varry,2)    
    E_a = edge(P,'canny',[tl th],sigma_varry(i)); 
    hold on
    subplot(2,3,i);colormap('gray');imshow(E_a);title(["Canny" "\sigma= " sigma_varry(i)])
end
subplot(2,3,6);colormap('gray');imshow(P);title(["Original"])
hold off

% 3.1 e)i)Varry tl
tl_varry=linspace(0.01,0.09,8);
figure( 'Position', [10 10 1440 960]);
for i = 1 : size(tl_varry,2)    
    E = edge(P,'canny',[tl_varry(i) th],sigma); 
    hold on
    subplot(2,size(tl_varry,2)/2,i);colormap('gray');imshow(E);title(["Canny" "tl= " tl_varry(i)])
end
hold off

%% Line Finding using Hough Transform
%3.2 b
theta = 0:180;
[H, xp] = radon(E,theta);

figure( 'Position', [10 10 900 350]);
subplot(1,2,1);imagesc(theta,xp,H),colormap(gca), colorbar;
title('R_{\theta} (X\prime)');
xlabel('\theta (degrees)')
ylabel('x''')
hold on;

%3.2 c
H_max = max(H,[],'all')
[y,x]=find(H == H_max)
hold off;

zoom=3
subplot(1,2,2);imagesc(theta(x-zoom:x+zoom),xp(y-zoom:y+zoom),H(y-zoom:y+zoom,x-zoom:x+zoom))
colormap(gca), colorbar
title({'Max R_{\theta} (X\prime) =' num2str(H_max),strcat("@ \theta= ", num2str(theta(x)), " and xp= ", num2str(xp(y)))});
xlabel('\theta (degrees)')
ylabel('x''')
hold on;
plot(theta(x),xp(y),'o',...
            'MarkerEdgeColor','red',...
            'MarkerFaceColor',[1 .6 .6])
        
[A, B] = pol2cart(theta(x)*pi/180, xp(y));
B=-B;
C=xp(y)^2+A*size(P,2)/2+B*size(P,1)/2;
y_line=@(x) -A/B.*x+C/B;
xl = 0,xr = size(P,2) - 1;
yl=y_line(xl)
yr=y_line(xr)
figure;colormap('gray');imshow(uint8(P));title("Macritchie (Finding Line)")
line([xl xr], [yl yr],'LineWidth',2);

%3.3 3D stereo 
clear all;
Pl = rgb2gray(imread('resource\corridorl.jpg'));
Pr = rgb2gray(imread('resource\corridorr.jpg'));
temp_size=11;
half=floor(temp_size/2);
whos Pl
whos Pr
figure;
subplot(1,2,1);colormap('gray');imshow(uint8(Pl));title("corridor left")
subplot(1,2,2);colormap('gray');imshow(uint8(Pr));title("corridor right")

map=disparity_map_Barn(Pl,Pr,11,11);
figure;imshow(-map,[-15 15]);title({"Disparity Map","Max D=15"})
%3.3 d
clear all;
Pl = rgb2gray(imread('resource\triclopsi2l.jpg'));
Pr = rgb2gray(imread('resource\triclopsi2r.jpg'));
temp_size=11;
half=floor(temp_size/2);
whos Pl
whos Pr
figure;
subplot(1,2,1);colormap('gray');imshow(uint8(Pl));title("corridor left")
subplot(1,2,2);colormap('gray');imshow(uint8(Pr));title("corridor right")

map=disparity_map_Barn(Pl,Pr,11,11);
figure;imshow(-map,[-15 15]);title({"Disparity Map","Max D=15"})

