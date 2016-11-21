%% load image
orig = imread('TestBoard1.jpg');

%orig = imgaussfilt(orig, 5);

%figure(1);
%imshow(orig);

%% crop just the board
img = double(rgb2gray(orig));
img = imgaussfilt(img, 5);

% decrease intensity around edges to avoid detecting wrong angle from edges
% outside the game board
[height,width] = size(img);
%t = linspace(-1,1,height);
%img = img .* repmat(exp(-(t(:)/0.99).^2),1,width);
%t = linspace(-1,1,width);
%img = img .* repmat(exp(-(t(:)'/0.99).^2),height,1);
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
[H, T, R] = hough(line_img, 'RhoResolution', 1, 'ThetaResolution', 1);

% find angle of board
% the resolution of the angle bins
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
    bin = floor(mod(peaks(i,2),90)/angle_accuracy) + 1;
    peak_counts(bin) = peak_counts(bin) + 1;
end
% get the most common angle (presumably the board)
[unused,board_angle] = max(peak_counts);
% assume the angle is in the middle of the angle_accuracy range
board_angle = floor((board_angle*angle_accuracy)-(angle_accuracy/2));

% find size of board
if board_angle-2*angle_accuracy < 0
    H1 = H(:,1:board_angle+2*angle_accuracy);
else
    H1 = H(:,board_angle-2*angle_accuracy:board_angle+2*angle_accuracy);
end
if board_angle+90+2*angle_accuracy > 180
    H2 = H(:,(board_angle+90)-2*angle_accuracy:180);
else
    H2 = H(:,(board_angle+90)-2*angle_accuracy:(board_angle+90)+2*angle_accuracy);
end
%H1 = sum(H1, 2);
%H2 = sum(H2, 2);
peaks1 = houghpeaks(H1, 19, 'Threshold', 0.3*max(H(:)));
max_peak1 = max(peaks1(:,1));
min_peak1 = min(peaks1(:,1));
peaks2 = houghpeaks(H2, 19, 'Threshold', 0.3*max(H(:)));
max_peak2 = max(peaks2(:,1));
min_peak2 = min(peaks2(:,1));

figure(1);
hold on
imshow(H, [], 'XData', T, 'YData', R, 'InitialMagnification', 'fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
plot([board_angle-90, board_angle-90,board_angle,board_angle],[R(max_peak1),R(min_peak1),R(max_peak2),R(min_peak2)],'s','color','white');
hold off

if R(max_peak1) > 0
    max_line1_intercept = [sind(board_angle-90)*R(max_peak1) cosd(board_angle-90)*R(max_peak1)];
else
    max_line1_intercept = [cosd(board_angle)*abs(R(max_peak1)) sind(board_angle)*R(max_peak1)];
end
max_line1_slope = -1/(max_line1_intercept(1)/max_line1_intercept(2));

if R(min_peak1) > 0
    min_line1_intercept = [sind(board_angle-90)*R(min_peak1) cosd(board_angle-90)*R(min_peak1)];
else
    min_line1_intercept = [cosd(board_angle)*abs(R(min_peak1)) sind(board_angle)*R(min_peak1)];
end
min_line1_slope = -1/(min_line1_intercept(1)/min_line1_intercept(2));

if R(max_peak2) > 0
    max_line2_intercept = [sind(board_angle)*R(max_peak2) cosd(board_angle)*R(max_peak2)];
else
    max_line2_intercept = [cosd(board_angle+90)*abs(R(max_peak2)) sind(board_angle+90)*R(max_peak2)];
end
max_line2_slope = -1/(max_line2_intercept(1)/max_line2_intercept(2));

if R(min_peak2) > 0
    min_line2_intercept = [sind(board_angle)*R(min_peak2) cosd(board_angle)*R(min_peak2)];
else
    min_line2_intercept = [cosd(board_angle+90)*abs(R(min_peak2)) sind(board_angle+90)*R(min_peak2)];
end
min_line2_slope = -1/(min_line2_intercept(1)/min_line2_intercept(2));

%average_slope = (min_line1_slope + max_line1_slope + 1/max_line2_slope + 1/min_line2_slope) / 4;

% this is a safety buffer put around the board to make sure nothing is cut
% off. It is a percentage of the height and width of the image
safety_buffer = .05;
%max_line1_intersect = max_line1(1).point1(2)-max_line1_slope*max_line1(1).point1(1) - safety_buffer*height;
max_line1_intersect = max_line1_intercept(1)-max_line1_slope*max_line1_intercept(2);
max_line1_intersect = max_line1_intersect - safety_buffer*height;
%min_line1_intersect = min_line1(1).point1(2)-min_line1_slope*min_line1(1).point1(1) + safety_buffer*height;
min_line1_intersect = min_line1_intercept(1)-min_line1_slope*min_line1_intercept(2);
min_line1_intersect = min_line1_intersect + safety_buffer*height;
%max_line2_intersect = max_line2(1).point1(2)-max_line2_slope*max_line1(1).point1(1) - safety_buffer*width;
max_line2_intersect = max_line2_intercept(1)-max_line2_slope*max_line2_intercept(2);
max_line2_intersect = max_line2_intersect + safety_buffer*width;
%min_line2_intersect = min_line2(1).point1(2)-min_line2_slope*min_line2(1).point1(1) + safety_buffer*width;
min_line2_intersect = min_line2_intercept(1)-min_line2_slope*min_line2_intercept(2);
min_line2_intersect = min_line2_intersect - safety_buffer*width;

figure(3);
imshow(img);
% horizontal
line([1,width],[max_line1_slope+max_line1_intersect, max_line1_slope*width+max_line1_intersect], 'Color', 'b');
line([1,width],[min_line1_slope+min_line1_intersect, min_line1_slope*width+min_line1_intersect], 'Color', 'b');
% vertical
if max_line2_slope == Inf
    %x = [max_line2(1).point1(1)+safety_buffer*width, max_line2(1).point1(1)+safety_buffer*width];
    x = [max_r2(1)+safety_buffer*width, max_r2(1)+safety_buffer*width];
    line([x(1), x(1)],[1,height],'Color','r');
else
    x = [(min_line1_intersect-max_line2_intersect)/(max_line2_slope-min_line1_slope),(max_line1_intersect-max_line2_intersect)/(max_line2_slope-max_line1_slope)];
    line([(1-max_line2_intersect)/max_line2_slope, (height-max_line2_intersect)/max_line2_slope],[1,height], 'Color', 'r');
end
if min_line2_slope == Inf
    %x = [[min_line2(1).point1(1)-safety_buffer*width, min_line2(1).point1(1)-safety_buffer*width] x];
    x = [[min_r2(1)-safety_buffer*width, min_r2(1)-safety_buffer*width] x];
    line([min_line2(1).point1(1)-safety_buffer*width, min_line2(1).point1(1)-safety_buffer*width],[1,height],'Color','r');
else
    x = [[(max_line1_intersect-min_line2_intersect)/(min_line2_slope-max_line1_slope),(min_line1_intersect-min_line2_intersect)/(min_line2_slope-min_line1_slope)] x];
    line([(1-min_line2_intersect)/min_line2_slope, (height-min_line2_intersect)/min_line2_slope],[1,height], 'Color', 'r');
end

% crop the image
img = orig;
y = [max_line1_slope*x(1)+max_line1_intersect,min_line1_slope*x(2)+min_line1_intersect,min_line1_slope*x(3)+min_line1_intersect,max_line1_slope*x(4)+max_line1_intersect];
board_height = ((y(2)-y(1))+(y(3)-y(4)))/2;
board_width = ((x(4)-x(1))+(x(3)-x(2)))/2;
mask = poly2mask([x x(1)],[y y(1)],height,width);
mask = imrotate(mask, board_angle);

%% convert image to grayscale
%  increase the importance of blue so that elements with more blue appear
%  lighter
%img = 0.2989 * img(:,:,1) + 0.5870 * img(:,:,2) + 0.1140 * img(:,:,3);
img = 0.0000 * img(:,:,1) + 0.0000 * img(:,:,2) + 0.9999 * img(:,:,3);
img = imrotate(img, board_angle);
img = uint8(img);
img_comp = imcomplement(img);

figure(4);
subplot(1,2,1);
imshow(img), gray(256);
subplot(1,2,2);
imshow(img_comp), gray(256);

%% correct for lighting before thresholding
light_corrected_img = imtophat(img, strel('disk', 120));
%light_corrected_img = imadjust(light_corrected_img,[0.3 0.7],[]);

figure(5);
imshow(light_corrected_img);

%% convert to gray level
img_black = im2bw(light_corrected_img, graythresh(img));
img_white = im2bw(light_corrected_img, 0.64);
img_black(mask==0) = 255;
img_white(mask==0) = 0;

figure(6);
subplot(1,2,1);
imshow(img_black);
subplot(1,2,2);
imshow(img_white);

% flip the black image to facilitate morphological image processing
img_black = imcomplement(img_black);

%% remove noise
% first try and remove any noise around the border from an inaccurate mask
% crop
struct_element = strel('rectangle', [100,20]);
img_black=img_black-imopen(img_black, struct_element);
img_white=img_white-imopen(img_white, struct_element);
struct_element = strel('rectangle', [20,100]);
img_black=img_black-imopen(img_black, struct_element);
img_white=img_white-imopen(img_white, struct_element);

struct_element = strel('disk', 20);

white_opening = imopen(img_white, struct_element);
black_opening = imopen(img_black, struct_element);

%%  erode to make sure pieces are sperated
struct_element = strel('disk', 10);

%white_opening = imerode(white_opening, struct_element);
black_opening = imerode(black_opening, struct_element);

figure(7);
subplot(1,2,1);
imshow(black_opening);
subplot(1,2,2);
imshow(white_opening);

%%  calculate the center of the pieces
black_opening = logical(black_opening);
white_opening = logical(white_opening);
regions_black = regionprops('table',black_opening,'Centroid','MajorAxisLength','MinorAxisLength');
regions_white = regionprops('table',white_opening,'Centroid','MajorAxisLength','MinorAxisLength');
centers_black = regions_black.Centroid;
centers_white = regions_white.Centroid;
diameters_black = mean([regions_black.MajorAxisLength regions_black.MinorAxisLength], 2);
radii_black = diameters_black / 2;
diameters_white = mean([regions_white.MajorAxisLength regions_white.MinorAxisLength], 2);
radii_white = diameters_white / 2;

figure(8);
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

figure(9);
imshow(img);
viscircles(centers_black, radii_black, 'Color', 'r');
viscircles(centers_white, radii_white, 'Color', 'b');

%%  overlay the grid
% the test grid will be fitted board_angle+-angle_offset
angle_offset = 5
