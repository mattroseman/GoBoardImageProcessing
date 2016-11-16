figure(1);
orig = imread('TestBoard1.jpg');
imshow(orig);
img = double(rgb2gray(orig));
img = imgaussfilt(img, 3);
filt_img = imfilter(img, h);

% normalize image
filt_img(filt_img < 0) = 0;
filt_img = filt_img - min(filt_img(:));
filt_img = filt_img / max(filt_img(:));

figure(2);
imshow(filt_img); gray(256);

% threshold image
img = im2bw(filt_img, 0.4);

figure(3);
imshow(img); gray(256); 

% Hough Transform
[H,T,R] = hough(img, 'RhoResolution', 0.5, 'ThetaResolution', 0.5);

figure(4);
imshow(H, [], 'XData', T, 'YData', R, 'InitialMagnification', 'fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;

%  find peaks in HOUGH
peaks = houghpeaks(H, 38, 'threshold', ceil(0.1*max(H(:))));
x = T(peaks(:,2)); y = R(peaks(:,1));
plot(x,y,'s','color','white');

%  find the lines
slopes = 90 - x;
[x, y] = pol2cart(x, y);
lines = [slopes, -slopes.*x + y];