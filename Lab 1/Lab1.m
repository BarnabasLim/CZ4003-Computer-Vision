%% 2.1 Contrast Stretching
% 2.1 Part a) 
clear all;
Pc = imread('resource\mrttrainbland.jpg');
whos Pc
P = rgb2gray(Pc);
whos P

%% 2.1 Contrast Stretching
% 2.1 Part b) 
figure;colormap('gray');imshow(uint8(P));title("Original Image")

% 2.1 Part c)
min_P=double(min(P(:)))%13
max_P=double(max(P(:)))%204

% 2.1 Part d) contrast stretching 
P2 = (double(P(:,:))-min_P).*(255/(max_P-min_P));

% 2.1 Part d) checking min max of P2
min(P2(:)), max(P2(:)) %0, 255

% 2.1 Part e) 
figure;colormap('gray');imshow(uint8(P2));title("Contrast Stretching")
%% 2.2 Histogtam Equilization
% 2.2 Part a) 
figure;imhist(P,10);title("Histogram with 10 bins (Original)")
figure;imhist(P,256);title("Histogram with 256 bins (Original)")

%Difference is that there are more bins. 
%For 10 bins, each bin contains grayscale of range 25, while
%for 256 bins, each bin contains grayscale of range 1.
%For 10 bins, the largers number of a bin ~5E+4, while
%for 10 bins, the largers number of a bin 2935.

% 2.2 Part b) Histogram Equalization
P3 = histeq(P,255);
figure;imhist(P3,256);title("Histogram with 256 bins(Histogram Equilization)")
figure;imhist(P3,10);title("Histogram with 10 bins(Histogram Equilization)")
figure;colormap('gray');imshow(uint8(P3));title("Histogram Equilized Image")

%Are the histograms equalized?
% Yes. The histogram are equilized as observed by the histogram being more
% evenly spread out from 0 to 255.
%What are the similarities and differences between the latter two histograms?
% The 10-bin histogram after histogram equalization  
% and the 256-bin histogram after histogram equalization 
% are similar in that both occupy the maximum range of gray level available from 0 to 255. 

% The main difference is regarding the frequency of each bin. 
% For 10-bin histogram (Figure 9), the frequency for each bin  is around the range of 1.5E+04 
% while  For 256-bin histogram (Figure 10), the frequency for each bin  is around the range of 500 to 3000. 

% Another difference is that spacing of each bin. 
% For 10-bin histogram (Figure 9), the bins are evenly spaced out while  
% for 256-bin histogram (Figure 10), the bins are more spaced out for grey levels between 0 to 150 
% when the frequency is high between 1000 to 3000 and 
% the bins are less spaced out for grey levels between 150 to 240 when the frequency is low at around 500.


% 2.2 Part c) Histogram Equalization Rerun
P3 = histeq(P,255);
figure;imhist(P3,256);title("Histogram with 256 bins (Histogram Equilization rerun)")
figure;colormap('gray');imshow(uint8(P3));title("Histogram Equilized image after rerun")
%Does the histogram become more uniform? Give suggestions as to why this occurs.
% No. In fact the histogram remains the same after redoing histogram
% equilisation. 
% This is because the bins have already been shifted based on their
% cumulative probabaily. Rerunning histogram equilisation will only
% allocate the bins to the same bins. 

%% 2.3 Linear Spatial Filtering
% 2.3 Part a) Gaussian Filters
h=@(X,Y,Sigma) exp(-(X.^2.+Y.^2)./(Sigma^2.*2))./(2*pi*Sigma^2);
sizes=5
x=-(sizes-1)/2:(sizes-1)/2;y=-(sizes-1)/2:(sizes-1)/2;
[X,Y] = meshgrid(x,y)

% 2.3 Part a)i) Gaussian Filters sigma = 1.0
h1=h(X,Y,1)
h1=h1./sum(h1,'all')
figure
mesh(X,Y,h1,'FaceAlpha','0.8'),title('Gaussian Filter ? = 1.0'),xlabel('X'),ylabel('Y'),zlabel('Z');

% 2.3 Part a)ii) Gaussian Filters sigma = 2.0
h2=h(X,Y,2)
h2=h2./sum(h2,'all')
figure
mesh(X,Y,h2,'FaceAlpha','0.8'),title('Gaussian Filter ? = 2.0'),xlabel('X'),ylabel('Y'),zlabel('Z');

% 2.3 Part b) Download the image ‘ntu-gn.jpg’
% 2.3 Part d) Download the image ‘ntu-sp.jpg’
ntu_gn = imread('resource\ntugn.jpg');
ntu_sp = imread('resource\ntusp.jpg');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uncomment desired image to apply gaussian filter
unfiltered_image=ntu_sp;
%unfiltered_image=ntu_gn;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
whos unfiltered_image
figure;colormap('gray');imshow(uint8(unfiltered_image))

% 2.3 Part c) Filter the image using the linear filters
% 2.3 Part d) Filter the image using the linear filters
unfiltered_image_h1=conv2(unfiltered_image, h1,'same');
figure;colormap('gray');imshow(uint8(unfiltered_image_h1))
unfiltered_image_h2=conv2(unfiltered_image, h2,'same');
figure;colormap('gray');imshow(uint8(unfiltered_image_h2))

% 2.3 Part c) Filter the image using the linear filters
%How effective are the filters in removing noise?
% The filters are effective at removing the gaussian noise. The higher the
% sigma the better the gaussian noise is removed.
% The filters can reduce speckled noise to some extend but is not very effective. 
% The higher the ? the better the speckled noise is removed.

%What are the trade-offs between using either of the two filters, or not filtering the image at all?
% The trade-off is that higher ? removes more gaussian noise but the more edges  blurred/ loss as a result.

% 2.3 Part e) 
%Are the filters better at handling Gaussian noise or speckle noise?
% The filters are better at handle Gaussian Noise compared to Speckled
% Noise
% Define Gaussian Noise and Speckled noise

%%  2.4 Median Filtering
% 2.3 Part b) Download the image ‘ntu-gn.jpg’
% 2.3 Part d) Download the image ‘ntu-sp.jpg’
ntu_gn = imread('resource\ntugn.jpg');
ntu_sp = imread('resource\ntusp.jpg');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uncomment desired image to apply Median filter
unfiltered_image=ntu_sp;
%unfiltered_image=ntu_gn;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
whos unfiltered_image
figure;colormap('gray');imshow(uint8(unfiltered_image))

% 2.3 Part c) Filter the image using the 3x3 Median
% 2.3 Part d) Filter the image using the 5x5 Median
unfiltered_image_median33=medfilt2(unfiltered_image,[3 3]);
figure;colormap('gray');imshow(uint8(unfiltered_image_median33))
unfiltered_image_median55=medfilt2(unfiltered_image,[3 3]);
figure;colormap('gray');imshow(uint8(unfiltered_image_median55))

% 2.3 Part c) Filter the image using the linear filters
%How effective are the 3x3 and 5x5 Median filters in removing gaussian noise?
% The filters are slightly effective at removing the gaussian noise but are good
% at removing the speckle noise. For Gaussian noise there is a slight
% imporvement when we move from a 3x3 to 5x5 median filter. 
% For Speckled Noise there is a huge improvement mainly because speckled
% noise tends to be white dots which are easily filtered by median filters.

%What are the trade-offs between using either of the two filters, or not filtering the image at all?
% Salt and pepper noise or speckled noise can be effectively removed from
% the images by 3x3 and 5x5 median filters. Edges are well preserved however noise that 
% with grey level that exhibit gaussian distribution will be less
% effectively removed because these noises tend to be closer to the median.

% 2.3 Part e) 
%Are the filters 3x3 and 5x5 median filter better at handling Gaussian noise or speckle noise?
% The x3 and 5x5 median filters are better at handling Speckled
% Noise then Gaussian Noise.


%% 2.5 Suppressing Noise Interference Patterns
% 2.5 Part a) Download the image ‘pck-int.jpg’

pck = imread('resource\pckint.jpg');
unfiltered_image=pck;
whos unfiltered_image
figure;colormap('gray');imshow(uint8(unfiltered_image));title("Original Image")

% 2.5 Part b) Obtain the Fourier transform F
F = fft2(pck);
F_manipulate=F;

% 2.5 Part b)compute the power spectrum S
S=abs(F).^2;
figure;imagesc(fftshift(S.^0.1));colormap('default');title("fft shift")
figure;imagesc(fftshift(S.^0.1));colormap('gray');title("fft shift (Grey)")

% 2.5 Part c) Redisplay the power spectrum without fftshift.
figure;imagesc(S.^0.1);colormap('default');title("fft")
figure;imagesc(S.^0.1);colormap('gray');title("fft (Grey)")
 
 % Top Right FFT Power Spectrum
 %u=[15, 19]
 %v=[247,251]
 
 % Bottom left FFT Power Spectrum
 %u=[239,243]
 %v=[7,11]
 
% 2.5 Part d) Set to zero the 5x5 neighbourhood elements of the peak
F(15:19,247:251)=0;
F(239:243,7:11)=0;
S=abs(F).^2;
figure;imagesc(fftshift(S.^0.1));colormap('default');title("fft shift filtered")
figure;imagesc(fftshift(S.^0.1));colormap('gray');title("fft shift filtered grey")

figure;imagesc(S.^0.1);colormap('default');title("fft filtered")
figure;imagesc(S.^0.1);colormap('gray');title("fft filtered (Grey)")

% 2.5 Part e) Compute the inverse Fourier transform using ifft2 and display the resultant image
figure;colormap('gray');imshow(uint8(ifft2(F)))
%Comment on the result and how this relates to step (c). Can you suggest any way to improve this? 
% As shown in Figure 35, by setting F(u,v)=0 for the two peaks, 
% we are able to remove most of the interference pattern. 
% This is because when we set F(u,v)=0 in step c 
% we are removing a specific sets of building block (‘atom’)  of the image 
% that corresponds to the interference patterns. 
% When we do an inverse Fourier Transform the specific atom with 
% “weight” =0 will have zero influence on the image. 

%Example of an atom pattern
M=256;N=256;
u=17;v=249;
[y_atom,x_atom]=meshgrid(0:M-1,0:N-1);
h=cos(2*pi*(u*x_atom/M+v*y_atom/N));
figure;imshow(uint8((h+1)*127));
%% Suppressing Noise Interference Patterns (2.5 Part e) 
%Attempts to improve 
% Extend filter V2
 % Top Right FFT Power Spectrum
 %u=[15, 19] ,[10,35],17
 %v=[247,251],249    ,[231,256]
 
 % Bottom left FFT Power Spectrum
 %u=[239,243],[223,248],241
 %v=[7,11]   ,9        ,[2,27]

F_manipulate(15:19,247:251)=0;
F_manipulate(239:243,7:11)=0;
F_manipulate(10:35,249)=0;
F_manipulate(223:248,9)=0;
F_manipulate(17,231:256)=0;
F_manipulate(241,2:27)=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uncomment section to apply different set of filters
%V3
% F_manipulate(15:19,247:251)=0;
% F_manipulate(239:243,7:11)=0;
% F_manipulate(10:256,249)=0;
% F_manipulate(1:248,9)=0;
% F_manipulate(17,1:256)=0;
% F_manipulate(241,2:256)=0;

%V4
% F_manipulate(15:19,247:251)=0;
% F_manipulate(239:243,7:11)=0;
% F_manipulate(1:256,249)=0;
% F_manipulate(1:256,9)=0;
% F_manipulate(17,1:256)=0;
% F_manipulate(241,1:256)=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S_manipulate=abs(F_manipulate).^2
figure;imagesc(fftshift(S_manipulate.^0.1));colormap('default');title("fft shift filtered")
figure;imagesc(fftshift(S_manipulate.^0.1));colormap('gray');title("fft shift filtered grey")

figure;imagesc(S_manipulate.^0.1);colormap('default');title("fft filtered")
figure;imagesc(S_manipulate.^0.1);colormap('gray');title("fft filtered (Grey)")
figure;colormap('gray');imshow(uint8(ifft2(F_manipulate)))

figure;colormap('gray');imshow(uint8(ifft2(F)))

%V5 and V6
x=-128:127;y=-128:127;
[xx yy] = meshgrid(x,y);
u = ones(256);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uncomment section to apply different set of filters
u((xx.^2+yy.^2)<17^2 & (xx.^2+yy.^2)>16^2 & abs(fftshift(F))>250)=0;  %V6
%u((xx.^2+yy.^2)<20^2 & (xx.^2+yy.^2)>15^2 & abs(fftshift(F))>250)=0;  %V5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;colormap('gray');imshow(uint8(u))

S_manipulate=fftshift(abs(F_manipulate).^0.1).^2.*u

figure;imagesc(S_manipulate);colormap('default');title("fft shift filtered")
figure;imagesc(S_manipulate);colormap('gray');title("fft shift filtered grey")

figure;imagesc(ifftshift(S_manipulate));colormap('default');title("fft filtered")
figure;imagesc(ifftshift(S_manipulate));colormap('gray');title("fft filtered (Grey)")

figure;colormap('gray');imshow(uint8(ifft2(ifftshift(fftshift(F_manipulate).*u))))
%% Suppressing Noise Interference Patterns (2.5 Part f) 
%2.5 Part f) attempt to “free” the primate by filtering out the fence
caged = imread('resource\primatecaged.jpg');
caged = rgb2gray(caged);
unfiltered_image=caged;
whos unfiltered_image
figure;colormap('gray');imshow(uint8(unfiltered_image));title("Original Image")

%Obtain the Fourier transform F
F_cage = fft2(caged);

%compute the power spectrum S
S_cage=abs(F_cage).^2;
figure;imagesc(fftshift(S_cage.^0.1));colormap('default');title("fft shift")
figure;imagesc(fftshift(S_cage.^0.1));colormap('gray');title("fft shift (Grey)")

%Redisplay the power spectrum without fftshift.
figure;imagesc(S_cage.^0.1);colormap('default');title("fft")
figure;imagesc(S_cage.^0.1);colormap('gray');title("fft (Grey)")

x=-128:127;y=-128:127;
[xx yy] = meshgrid(x,y);
u1 = ones(256);
u2 = ones(256);
u3 = ones(256);
u4 = ones(256);
u5 = ones(256);

u1(4:8,246:248)=0;
u1(250:254,10:12)=0;

u2(14:17,247:249)=0;
u2(241:244,9:11)=0;

u3(9:11,234:239)=0;
u3(247:249,19:24)=0;

u4(20:22,235:239)=0;
u4(236:238,18:23)=0;

u5(14:16,223:227)=0;
u5(242:244,31:35)=0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uncomment section to apply different set of filters
% F_cage_filtered=F_cage.*u1;
% F_cage_filtered=F_cage.*u1.*u2;
% F_cage_filtered=F_cage.*u1.*u2.*u3;
% F_cage_filtered=F_cage.*u1.*u2.*u3.*u4;
F_cage_filtered=F_cage.*u1.*u2.*u3.*u4.*u5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S_cage=abs(F_cage_filtered).^2;
figure;imagesc(fftshift(S_cage.^0.1));colormap('default');title("fft shift filtered")
figure;imagesc(fftshift(S_cage.^0.1));colormap('gray');title("fft shift filtered grey")

figure;imagesc(S_cage.^0.1);colormap('default');title("fft filtered")
figure;imagesc(S_cage.^0.1);colormap('gray');title("fft filtered (Grey)")

% 2.5 Part e) Compute the inverse Fourier transform using ifft2 and display the resultant image
figure;colormap('gray');imshow(uint8(ifft2(F_cage_filtered)))




%%
% 2.6 Part a) 
P=imread('resource\book.jpg');
figure;imshow(P)

% 2.6 Part b)
%selection begins Top left rotate clockwise
[X Y] = ginput(4);

% 2.6 Part c)
%begins Top left rotate clockwise
X_im=[0,210,210,0];
Y_im=[0,0,297,297];


A=[X(1), Y(1), 1, 0, 0, 0, -(X_im(1)*X(1)), -(X_im(1)*Y(1));
   0, 0, 0, X(1), Y(1), 1, -(Y_im(1)*X(1)), -(Y_im(1)*Y(1));
   X(2), Y(2), 1, 0, 0, 0, -(X_im(2)*X(2)), -(X_im(2)*Y(2));
   0, 0, 0, X(2), Y(2), 1, -(Y_im(2)*X(2)), -(Y_im(2)*Y(2));
   X(3), Y(3), 1, 0, 0, 0, -(X_im(3)*X(3)), -(X_im(3)*Y(3));
   0, 0, 0, X(3), Y(3), 1, -(Y_im(3)*X(3)), -(Y_im(3)*Y(3));
   X(4), Y(4), 1, 0, 0, 0, -(X_im(4)*X(4)), -(X_im(4)*Y(4));
   0, 0, 0, X(4), Y(4), 1, -(Y_im(4)*X(4)), -(Y_im(4)*Y(4));
   ];

v=[X_im(1);Y_im(1);X_im(2);Y_im(2);X_im(3);Y_im(3);X_im(4); Y_im(4)];

u = A \ v;
size(u)% 8 x 1

U = reshape([u;1], 3, 3)'; % Reshape fills columns first
size(U)% 3 x 3
%U =
% 1.4324    1.4858 -246.3794
%-0.4302    3.5491  -33.8779
% 0.0001    0.0050    1.0000

w = U*[X'; Y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:))

% 2.6 Part d)
T = maketform('projective', U');
P2 = imtransform(P, T, 'XData', [0 210], 'YData', [0 297]);
figure;imshow(P2)

