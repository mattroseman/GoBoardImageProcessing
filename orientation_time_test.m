img = imread('TestBoard5.jpg');
img = double(rgb2gray(img));

disp('running fourier transform');

tic;
PS = fftshift(2*log(abs(fft2(img))+1));
M = max(PS(:));
fourier = uint8(255*(PS/M));
toc;

disp('running fourier transform with imresize');

tic;
PS = fftshift(2*log(abs(fft2(imresize(img, [507 NaN])))+1));
M = max(PS(:));
fourier = uint8(255*(PS/M));
toc;

disp('running hough transform');

tic;
img = imgaussfilt(img, 5);
img(img<0) = 0;
img = img - min(img(:));
img = img / max(img(:));
h = [1 1 1 1 1; 1 1 1 1 1; 1 1 -24 1 1; 1 1 1 1 1; 1 1 1 1 1];
line_img = imfilter(img, h);
line_img(line_img < 0) = 0;
line_img = line_img - min(line_img(:));
line_img = line_img / max(line_img(:));
line_img = im2bw(line_img, 0.4);
[H, T, R] = hough(line_img, 'RhoResolution', 5, 'ThetaResolution', 1);
toc;
