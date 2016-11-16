%% load image
orig = imread('TestBoard1.jpg');

%% scale image down
%orig = imgaussfilt(orig, 5);

figure(1);
imshow(orig);

%% crop just the board
img = double(rgb2gray(orig));

% decrease intensity around edges to avoid detecting wrong angle
[height,width] = size(img);
t = linspace(-1,1,height);
img = img .* repmat(exp(-(t(:)/0.9).^2),1,width);
t = linspace(-1,1,width);
img = img .* repmat(exp(-(t(:)'/0.9).^2),height,1);
% normalize
img(img<0) = 0;
img = img - min(img(:));
img = img / max(img(:));

h = [1 1 1 1 1; 1 1 1 1 1; 1 1 -24 1 1; 1 1 1 1 1; 1 1 1 1 1];
line_img = imfilter(img, h);
% normalize
line_img(line_img < 0) = 0;
line_img = line_img - min(line_img(:));
line_img = line_img / max(line_img(:));
% threshold
line_img = im2bw(line_img, 0.4);
% Hough transform
[H, T, R] = hough(line_img, 'RhoResolution', 0.5, 'ThetaResolution', 1);

figure(2);
imshow(H, [], 'XData', T, 'YData', R, 'InitialMagnification', 'fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;

% find angle of board
% the degree of error for the angle of the board
angle_accuracy = 3;
num_peaks = 20;
% bins to store the number of peaks found at each possible board rotation
% index 1 stores peaks found from 0 deg to angle_accuracy and 90 degrees to
% 90+angle_accuracy. last index stores peaks found from 90-angle_accuracy
% to 89 or 180-angle_accuracy to 179 
% (or whatever the remainder of 90/angle_accuracy)
peak_counts = zeros(1, ceil(90/angle_accuracy));
peaks = houghpeaks(H, num_peaks);
for i = 1:length(peaks)
    % modulus 90 because we know the board is at a 90 degree angle, so we
    % combine count of peaks found at 0-90 and 90-180. (for instance a peak
    % found at 20 deg and 110 deg would go in the same peak_counts bin
    bin = floor(mod(peaks(i,2),90)/3) + 1;
    peak_counts(bin) = peak_counts(bin) + 1;
end
% get the most common angle (presumably the board)
[unused,board_angle] = max(peak_counts);
% assume the angle is in the middle of the angle_accuracy range
board_angle = (board_angle*angle_accuracy)-(angle_accuracy/2);
plot([board_angle-90, board_angle],0,'s','color','white');

img = orig;
%  increase the importance of blue so that elements with more blue appear
%  lighter
%img = 0.2989 * img(:,:,1) + 0.5870 * img(:,:,2) + 0.1140 * img(:,:,3);
img = 0.0000 * img(:,:,1) + 0.0000 * img(:,:,2) + 0.9999 * img(:,:,3);
img = uint8(img);
img_comp = imcomplement(img);

figure(3);
subplot(1,2,1);
imshow(img), gray(256);
subplot(1,2,2);
imshow(img_comp), gray(256);

%% correct for lighting before thresholding
light_corrected_img = imtophat(img, strel('disk', 80));

figure(4);
imshow(light_corrected_img);

%% convert to gray level
img_black = im2bw(img, graythresh(img));
img_white = im2bw(img, 0.62);

figure(5);
subplot(1,2,1);
imshow(img_black);
subplot(1,2,2);
imshow(img_white);

% flip the black image to facilitate morphological image processing
img_black = imcomplement(img_black);

%% remove noise
struct_element = strel('disk', 25);

white_opening = imopen(img_white, struct_element);
black_opening = imopen(img_black, struct_element);

%%  erode to make sure pieces are sperated
struct_element = strel('disk', 15);

white_opening = imerode(white_opening, struct_element);
black_opening = imerode(black_opening, struct_element);

figure(6);
subplot(1,2,1);
imshow(black_opening);
subplot(1,2,2);
imshow(white_opening);

%%  calculate the center of the pieces
regions_black = regionprops('table',black_opening,'Centroid','MajorAxisLength','MinorAxisLength');
regions_white = regionprops('table',white_opening,'Centroid','MajorAxisLength','MinorAxisLength');
centers_black = regions_black.Centroid;
centers_white = regions_white.Centroid;
diameters_black = mean([regions_black.MajorAxisLength regions_black.MinorAxisLength], 2);
radii_black = diameters_black / 2;
diameters_white = mean([regions_white.MajorAxisLength regions_white.MinorAxisLength], 2);
radii_white = diameters_white / 2;

figure(7);
imshow(img);
viscircles(centers_black, radii_black, 'Color', 'r');
viscircles(centers_white, radii_white, 'Color', 'b');

%  put the radiuses in bins and find the most common range
% combine black and white radii
radii = vertcat(radii_black, radii_white);
% the bin size in pixels
radius_accuracy = 20;
% each index of radii counts is a bin. index 1 is a bin for
% 1 to radius_accuracy, and the last index is a bin for
% max(radii)-radius_accuracy+1 to max(radii) or whatever is left after the
% last full bin
radii_counts = zeros(1, ceil(max(radii)/radius_accuracy)+1);
for i = 1:length(radii)
    bin = ceil(radii(i)/radius_accuracy);
    radii_counts(bin) = radii_counts(bin) + 1;
end
[unused, avg_stone_radii] = max(radii_counts);
avg_stone_radii = avg_stone_radii*radius_accuracy;
centers_black = centers_black(radii_black <= avg_stone_radii+radius_accuracy & radii_black > avg_stone_radii-radius_accuracy, :);
centers_white = centers_white(radii_white <= avg_stone_radii+radius_accuracy & radii_white > avg_stone_radii-radius_accuracy, :);

figure(8);
imshow(img);
viscircles(centers_black, radii_black, 'Color', 'r');
viscircles(centers_white, radii_white, 'Color', 'b');

%%  overlay the grid
board_angle
