%% load image
orig = imread('TestBoard5.jpg');

%% calculate Hough transform

% convert image to grayscale
img = double(rgb2gray(orig));
% apply a gaussian filter to aid Hough transform
img = imgaussfilt(img, 5);

% decrease intensity around edges to avoid detecting wrong angle from edges
% outside the game board
% this assumes the board is relatively center in the image
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
% RhoResolution can be decreased to increase accuracy of board dimensions
% ThetaResolution can be decreased to increase accuracy of board dimensions
[H, T, R] = hough(line_img, 'RhoResolution', 5, 'ThetaResolution', 1);

%% find angle of board

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

%% find size of board

% the range used is 2*angle accuracy to account for different angles of
% the edges on either side of the go-board, due to perspective

% if the angle_accuracy range extends beyond 0 degrees then just look
% starting at 0 degrees
% Note - could flip the second half of the Hough transform and put in the
% beginning to include this lost data
if board_angle-2*angle_accuracy < 0
    H1 = H(:,1:board_angle+2*angle_accuracy);
else
    H1 = H(:,board_angle-2*angle_accuracy:board_angle+2*angle_accuracy);
end
% if the angle_accuracy range extends beyond 180 degrees then just look
% ending at 180 degrees
% Note - same process as noted above could be use to include the lost data
if board_angle+90+2*angle_accuracy > 180
    H2 = H(:,(board_angle+90)-2*angle_accuracy:180);
else
    H2 = H(:,(board_angle+90)-2*angle_accuracy:(board_angle+90)+2*angle_accuracy);
end

% NOTE - throughout this program the notation max_foo1, min_foo1, max_foo2,
% and min_foo2 is used to indicate different data of the boards size
% max_foo1 should indicate the top of the board
% min_foo1 should indicate the bottom of the board
% max_foo2 should indicate the right of the board
% min_foo2 should indicate the left of the board

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

% convert the 4 peaks from polar coordinates to the cartesian lines they
% represent (y = mx + b)

% if rho is positive
if R(max_peak1) > 0
    max_line1_intercept = [sind(board_angle-90)*R(max_peak1) cosd(board_angle-90)*R(max_peak1)];
% if rho is negative
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
% off. The lines previously calculated have a high chance of being along
% the outer gridlines, which would leave half the pieces along these grid
% lines cut off.
horz_safety_buffer = 200;
vert_safety_buffer = 200;

max_line1_intersect = max_line1_intercept(1)-max_line1_slope*max_line1_intercept(2);
max_line1_intersect = max_line1_intersect - horz_safety_buffer;

min_line1_intersect = min_line1_intercept(1)-min_line1_slope*min_line1_intercept(2);
min_line1_intersect = min_line1_intersect + horz_safety_buffer;

max_line2_intersect = max_line2_intercept(1)-max_line2_slope*max_line2_intercept(2);
max_line2_intersect = max_line2_intersect + vert_safety_buffer;

min_line2_intersect = min_line2_intercept(1)-min_line2_slope*min_line2_intercept(2);
min_line2_intersect = min_line2_intersect - vert_safety_buffer;

figure(3);
imshow(img);
hold on

% horizontal
line([1,width],[max_line1_slope+max_line1_intersect, max_line1_slope*width+max_line1_intersect], 'Color', 'b');
line([1,width],[min_line1_slope+min_line1_intersect, min_line1_slope*width+min_line1_intersect], 'Color', 'r');

% vertical
% if the vertical lines have an infinite slope then, calculation errors
% will arise from calculating with the slope
if max_line2_slope == Inf
    x = [max_r2(1)+vert_safety_buffer, max_r2(1)+vert_safety_buffer];
    line([x(1), x(1)],[1,height],'Color','g');
else
    x = [(min_line1_intersect-max_line2_intersect)/(max_line2_slope-min_line1_slope),(max_line1_intersect-max_line2_intersect)/(max_line2_slope-max_line1_slope)];
    line([(1-max_line2_intersect)/max_line2_slope, (height-max_line2_intersect)/max_line2_slope],[1,height], 'Color', 'g');
end
if min_line2_slope == Inf
    x = [[min_r2(1)-vert_safety_buffer, min_r2(1)-vert_safety_buffer] x];
    line([min_line2(1).point1(1)-safety_buffer*width, min_line2(1).point1(1)-safety_buffer*width],[1,height],'Color','y');
else
    x = [[(max_line1_intersect-min_line2_intersect)/(min_line2_slope-max_line1_slope),(min_line1_intersect-min_line2_intersect)/(min_line2_slope-min_line1_slope)] x];
    line([(1-min_line2_intersect)/min_line2_slope, (height-min_line2_intersect)/min_line2_slope],[1,height], 'Color', 'y');
end
hold off

%% crop the image

img = orig;
y = [max_line1_slope*x(1)+max_line1_intersect,min_line1_slope*x(2)+min_line1_intersect,min_line1_slope*x(3)+min_line1_intersect,max_line1_slope*x(4)+max_line1_intersect];
board_height = ((y(2)-y(1))+(y(3)-y(4)))/2;
board_width = ((x(4)-x(1))+(x(3)-x(2)))/2;
mask = poly2mask([x x(1)],[y y(1)],height,width);
mask = imrotate(mask, board_angle);

%% convert image to grayscale

% Along the rgb dimension black has all low values, white all hight, and
% the board (woodish brown?) has high red and green, but medium blue
% to better contrast the white pieces from the board I just use the blue
% values when converting to grayscale
% original conversion ratios
%img = 0.2989 * img(:,:,1) + 0.5870 * img(:,:,2) + 0.1140 * img(:,:,3);
img = 0.0000 * img(:,:,1) + 0.0000 * img(:,:,2) + 0.9999 * img(:,:,3);
% this centers the grid within the image, making later operations much
% easier
img = imrotate(img, board_angle);
[height,width] = size(img);
img = uint8(img);

figure(4);
subplot(1,2,1);
imshow(img), gray(256);
subplot(1,2,2);
imshow(imcomplement(img)), gray(256);

%% correct for lighting before thresholding
% the radius of this structuring element should be larger than the
% go-pieces radius
light_corrected_img = imtophat(img, strel('disk', 75));
%light_corrected_img = imadjust(light_corrected_img,[0.3 0.7],[]);

figure(5);
imshow(light_corrected_img);

%% convert to binary level
% this threshold should be raised if not enough black pieces remain
img_black = im2bw(img, 0.30);
% this threshold should be lowered if not enough white pieces remain
img_white = im2bw(img, 0.70);
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

% the radius of this structuring element needs to be less than the radius
% of the pieces so they are not eroded away during the erosion step of
% imopen

struct_element = strel('disk', 16);
img_black = imopen(img_black, struct_element);

struct_element = strel('disk', 16);
img_white = imopen(img_white, struct_element);

%%  erode to make sure pieces are sperated

% if pieces are touching on the go board, opening will not help then
% disconnect. This needs to happen in order to detect that they are
% seperate pieces. Erosion is performed to seperate them.

% the diameter of this structuring element should be wider than the width
% of the connection between two pieces, while smaller than the diameter of
% the actual pieces
struct_element = strel('disk', 25);

img_white = imerode(img_white, struct_element);
img_black = imerode(img_black, struct_element);

% try and remove any noise around the border from an inaccurate mask
% crop. Also if any connected pieces remain, this should disconnect them

% rectangle needs to be longer than the pieces diameter but shorter than
% 2*diameter. This is so the pieces themselves aren't cut in half, but if
% there are still connected pieces, the connection will be removed.
% The width should be enough such that if two pieces are
% connected, the connection should be removed, but there will still be a
% connection so the individual pieces remain single regions and aren't cut
% in half. This is because of the curve of the pieces. As the structuring
% element approaches the edge of one of the connected pieces, at some point
% the width should become longer than the cross section height/width of the
% piece. Thus keeping a connection between the two piece halfs.
struct_element = strel('rectangle', [100,15]);
img_black=img_black-imopen(img_black, struct_element);
img_white=img_white-imopen(img_white, struct_element);
struct_element = strel('rectangle', [15,100]);
img_black=img_black-imopen(img_black, struct_element);
img_white=img_white-imopen(img_white, struct_element);

figure(7);
subplot(1,2,1);
imshow(img_black);
subplot(1,2,2);
imshow(img_white);

%%  calculate the center of the pieces

img_black = logical(img_black);
img_white = logical(img_white);
regions_black = regionprops('table',img_black,'Centroid','MajorAxisLength','MinorAxisLength');
regions_white = regionprops('table',img_white,'Centroid','MajorAxisLength','MinorAxisLength');
centers_black = regions_black.Centroid;
centers_white = regions_white.Centroid;
% regionprops essentially fits the smallest elipses such that the entire
% region is inside of it. The diameter of the pieces should be about the
% average of the elipses major and minor axis.
diameters_black = mean([regions_black.MajorAxisLength regions_black.MinorAxisLength], 2);
radii_black = diameters_black / 2;
diameters_white = mean([regions_white.MajorAxisLength regions_white.MinorAxisLength], 2);
radii_white = diameters_white / 2;

% put the radiuses in bins and find the mode range

%% eliminate regions too small or too large

% combine black and white radii
radii = vertcat(radii_black, radii_white);
% the bin size in pixels
radius_accuracy = 75;
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
radii_black = radii_black(radii_black <= avg_stone_radii+radius_accuracy & radii_black > avg_stone_radii-radius_accuracy);
centers_white = centers_white(radii_white <= avg_stone_radii+radius_accuracy & radii_white > avg_stone_radii-radius_accuracy, :);
radii_white = radii_white(radii_white <= avg_stone_radii+radius_accuracy & radii_white > avg_stone_radii-radius_accuracy);

figure(8);
imshow(img);
hold on
viscircles(centers_black, radii_black, 'Color', 'r');
viscircles(centers_white, radii_white, 'Color', 'b');
hold off

centers = [centers_black; centers_white];

%% extrapolate grid

topmost_piece = min(centers(:,2));
botmost_piece = max(centers(:,2));
leftmost_piece = min(centers(:,1));
rightmost_piece = max(centers(:,1));
center = [(botmost_piece-topmost_piece)/2 (rightmost_piece-leftmost_piece)/2];

x_values = linspace(leftmost_piece,rightmost_piece,19);
y_values = linspace(topmost_piece,botmost_piece,19);

figure(9);
imshow(imrotate(orig,board_angle));
hold on
for i=1:19
    line([1,width],[y_values(i),y_values(i)],'Color','g');
    line([x_values(i),x_values(i)],[1,height],'Color','g');
end
plot(centers_black(:,1),centers_black(:,2),'ro');
plot(centers_white(:,1),centers_white(:,2),'bo');
hold off