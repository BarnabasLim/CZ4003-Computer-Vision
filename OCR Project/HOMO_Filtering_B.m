function Ihmf =HOMO_Filtering_B(gray)
%https://blogs.mathworks.com/steve/2013/06/25/homomorphic-filtering-part-1/
I = gray;
%imshow(I)
I = im2double(I);
I = log(1 + I);

M = 2*size(I,1) + 1;
N = 2*size(I,2) + 1;

%Creating High Pass Filter
sigma = 10;

[X, Y] = meshgrid(1:N,1:M);
centerX = ceil(N/2);
centerY = ceil(M/2);
gaussianNumerator = (X - centerX).^2 + (Y - centerY).^2;
H = exp(-gaussianNumerator./(2*sigma.^2));
H = 1 - H;
figure( 'Position', [10 10 900 600]);
subplot(1,2,1);imshow(H);title('High Pass Filter \sigma =10')
subplot(1,2,2);mesh(X,Y,H,'FaceAlpha','0.8'),title('High Pass Filter \sigma =10'),xlabel('X'),ylabel('Y'),zlabel('Z');

H = fftshift(H);

If = fft2(I, M, N);

%Pass fft2 image through High Pass Filter
Iout = real(ifft2(H.*If));
Iout = Iout(1:size(I,1),1:size(I,2));

Ihmf = exp(Iout) - 1;
end