%%

clear; clc; close all;

% Function to compress one block
function img_approx = compress_block(block, k)

    DCT_coeff = dct2(block);    % Pixles --> DCT coefficients

    [~, idx] = sort(abs(DCT_coeff(:)), 'descend');  % sort coefficients
    idx = idx(1:k); % Index of k largest coefficients

    new_block = zeros(size(DCT_coeff));     % Make matrix of zeroes 
    new_block(idx) = DCT_coeff(idx);        % Set top k pixles to original values, rest to 0

    img_approx = idct2(new_block);      % Reverse DCT transform
end


% image_file = 'bilder/rice.tif'
% blocksize = size of each block
% k = Number of DCT coeff to save per block

function compress_img(image_file, blocksize, k)    % Function to compress image

    img = im2double(imread(image_file));     % read image and normalize
    
    % Check that image is grayscale and divisable by blocksize
    assert(ismatrix(img), 'Input must be a grayscale image.'); 
    assert(all(mod(size(img), blocksize) == 0), sprintf('Image size not divisable by %d.',blocksize));  
    
    % Process every block and put output image together
    img_approx = blockproc(img, [blocksize blocksize], @(block) compress_block(block.data, k));
    
    % Clip reconstructed values to the image range
    img_approx_clipped = min(max(img_approx, 0), 1);
    
    % Plot images
    figure

    subplot(1, 2, 1);
    imshow(img, [0 1]);
    title('Original');
    
    subplot(1, 2, 2);
    imshow(img_approx_clipped, [0 1]);
    title(sprintf('DCT with %d blocks, top %d coeff/block', blocksize, k));
end

image_file = 'bilder/cameraman.tif';
blocksize = 8; 
k = 8;
compress_img(image_file, blocksize, k)