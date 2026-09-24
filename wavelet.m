%% Read and display image data
img = imread('trees.tif');
if size(img, 3) == 3
    img = rgb2gray(img);
end
img = im2double(img);
%% Wavelet compression 
max_level = wmaxlev(size(img),'haar')