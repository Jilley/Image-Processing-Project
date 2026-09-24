%% First stage
clear; clc; close all

image = im2double(imread('bilder/cameraman.tif'));  % load image
imshow(image)   % show image

block_size = 8;

len_x = size(image, 2);   % number of columns: 256
len_y = size(image, 1);   % number of rows: 256

number_boxes_x = len_x/block_size;
number_boxes_y = len_y/block_size;

x_starts = zeros(number_boxes_x);
y_starts = zeros(number_boxes_y);

for i = 1:number_boxes_x
    index = (i-1)*block_size + 1;
    x_starts(i) = index;
end

for i = 1:number_boxes_y
    index = (i-1)*block_size + 1;
    y_starts(i) = index;
end

number_boxes = number_boxes_x * number_boxes_y;


[height, width] = size(image);

% Pad the right and bottom so arbitrary image sizes work
padded_height = ceil(height / block_size) * block_size;
padded_width  = ceil(width  / block_size) * block_size;

I_padded = zeros(padded_height, padded_width);
I_padded(1:height, 1:width) = image;

blocks = mat2cell(I_padded, ...
    repmat(block_size, 1, padded_height / block_size), ...
    repmat(block_size, 1, padded_width  / block_size));

% Example: the block in row 1, column 2
imshow(blocks{1, 2});