clear all;
close all;
clc;
%path of the image directory
path = "C:\Users\Shiva Kumar Dande\Desktop\Semester 2\CV_Project\Project 2_Image_mosaicing\DanaHallWay1";
%reading and displaying images
input_images = read_images_from_path(path);
% display_images(input_images, 'Colour  Images');
%converting images to gray scale
gray_images = convert_grayScale(input_images);
TFORM = [];
numImages = size(input_images, 4);
imageSize = []; %zeros(numImages, 2);
wraped_image = gray_images(:, :, :, 1);


    [r1, c1] = harris_corner_alg(wraped_image);
    [r2, c2] = harris_corner_alg(gray_images(:, :, :, i+1));
    row_col1 = [r1, c1];
    row_col2 = [r2, c2];
    [max_ncc_values, matched_corner_sets] = norm_cross_corr(row_col1, wraped_image, row_col2, gray_images(:, :, :, i+1));
    ncc_threshold = 0.75;
    [ncc_th, matched_corner_sets_th] = threshold_ncc(max_ncc_values, matched_corner_sets, ncc_threshold);
    corners_image1 = flip(matched_corner_sets(:, 1:2), 2);
    corners_image2 = flip(matched_corner_sets(:, 3:4), 2);
    figure;
    showMatchedFeatures(wraped_image, gray_images(:, :, :, i+1), corners_image1, corners_image2, 'montage');
    title('Corner matches between the two images');
    waitforbuttonpress;
    close;
    corners_image1_th = flip(matched_corner_sets_th(:, 1:2), 2);
    corners_image2_th = flip(matched_corner_sets_th(:, 3:4), 2);
    figure;
    showMatchedFeatures(wraped_image, gray_images(:, :, :, i+1), corners_image1_th, corners_image2_th, 'montage');
    title('Corner matches between the two images after thresholding the NCC Values');
    waitforbuttonpress;
    close;
    [inlier_sets, tform] = RANSAC(matched_corner_sets_th);
%     tform.T
%     TFORM(:, :, i) = tform.T';
    figure;
    showMatchedFeatures(wraped_image, gray_images(:, :, :, i+1), flip(inlier_sets(:, 1:2), 2), flip(inlier_sets(:, 3:4), 2), 'montage');
    title('Corner matches between the two images after RANSAC');
    waitforbuttonpress;
    close;
    wraped_image = wrap_images(wraped_image, gray_images(:, :, :, i), tform);


figure;
imshow(wraped_image);
title("Final Output");

function img_combined = wrap_images(img1, img2, H)
        homography = H.T';
        homography([1 2], :) = homography([2 1], :);
        homography(:, [1 2]) = homography(:, [2 1]);
        % Compute dimensions of the resulting image
        [w1, h1, ~] = size(img1);
        [w2, h2, ~] = size(img2);
        corners = single([0, 0; 0, h2; w2, h2; w2, 0]);
        corners_transformed = transformPointsForward(H, corners);
        x_min = min(min(corners_transformed(:, 1)), 0);
        x_max = max(max(corners_transformed(:, 1)), w1);
        y_min = min(min(corners_transformed(:, 2)), 0);
        y_max = max(max(corners_transformed(:, 2)), h1);
        output_shape = [round(x_max - x_min), round(y_max - y_min)];
        % Compute translation matrix to shift the transformed image into place
        translation_matrix = [1, 0, -y_min; 0, 1, -x_min; 0, 0, 1];

        % Warp images using the homography matrix and translation matrix
        img1_warped = imwarp(img1, projective2d((translation_matrix * homography)'), 'OutputView', imref2d(output_shape));
        img2_warped = imwarp(img2, projective2d(translation_matrix'), 'OutputView', imref2d(output_shape));
        img_combined = max(img1_warped, img2_warped);
        figure;
        imshow(img_combined);
        title('Combined Image (2 Images)');
        waitforbuttonpress;
        close;
end

function homography_matrix = compute_homography(src, dst, i)  
   
    x1 = src(1,1); y1 = src(1,2);
    x2 = dst(1,1); y2 = dst(1,2);
    x3 = src(2,1); y3 = src(2,2);
    x4 = dst(2,1); y4 = dst(2,2);
    x5 = src(3,1); y5 = src(3,2);
    x6 = dst(3,1); y6 = dst(3,2);
    x7 = src(4,1); y7 = src(4,2);
    x8 = dst(4,1); y8 = dst(4,2);
   
    % Construct the matrix A
    A = [-x1, -y1, -1, 0, 0, 0, x2*x1, y2*x1, x2;
     0, 0, 0, -x1, -y1, -1, x2*y1, y2*y1, y2;
     -x3, -y3, -1, 0, 0, 0, x4*x3, y4*x3, x4;
     0, 0, 0, -x3, -y3, -1, x4*y3, y4*y3, y4;
     -x5, -y5, -1, 0, 0, 0, x6*x5, y6*x5, x6;
     0, 0, 0, -x5, -y5, -1, x6*y5, y6*y5, y6;
     -x7, -y7, -1, 0, 0, 0, x8*x7, y8*x7, x8;
     0, 0, 0, -x7, -y7, -1, x8*y7, y8*y7, y8];
    % Compute the null space of A using singular value decomposition
    [U, S, V] = svd(A);
    h = U(:,end);
    h(end+1) = 1;
    homography_matrix = reshape(h,3,3)';
    if (i == 1)
        h = V(:, end);
        homography_matrix = reshape(h,3,3)';
    end
end
function predicted_dst_points = apply_homography(homography_matrix, src_points)
    H = homography_matrix;
    predicted_dst_points = transformPointsForward(projective2d(H'), src_points);
end

function num_inliers = compute_inliers(dst_points, predicted_dst, maxDistance)
    distances = sqrt(sum((predicted_dst - dst_points).^2, 2));
    % find inliers based on distance threshold
    num_inliers = sum((distances < maxDistance));
end

function tform = compute_least_sq_homography(src_pts,dst_pts)
    n = size(src_pts,1);
    A = zeros(2*n,9); 
    tform = zeros(3);
    for i = 1:n
        xsi = src_pts(i,1);
        ysi = src_pts(i,2);
        xdi = dst_pts(i,1);
        ydi = dst_pts(i,2);
        A((2*i)-1,:) = [xsi,ysi,1,0,0,0,-(xdi*xsi),-(xdi*ysi),-(xdi)];
        A(2*i,:) = [0,0,0,xsi,ysi,1,-(ydi*xsi),-(ydi*ysi),-(ydi)];
    end
    [~, ~, V] = svd(A);
    h = V(:, end);
    tform = reshape(h,3,3)';
    tform = tform/ tform(3,3);
end
function [inlier_sets, tform] = RANSAC(matched_corner_sets_th)
    iterations = 72;
    maxDistance = 20;
    max_inliers = 0;
    src_points = [matched_corner_sets_th(:, 1:2)];
    dst_points = [matched_corner_sets_th(:, 3:4)];
%     [homography_matrix, inliers] = estimateGeometricTransform(points1, points2, 'projective', 'MaxNumTrials', 2000, 'Confidence', 99);
    for i = 1 : iterations
        indices = randperm(size(matched_corner_sets_th, 1), 4);
        points = matched_corner_sets_th(indices, :);
        src = [points(:, 1:2)];
        dst = [points(:, 3:4)];
        homography_matrix = compute_homography(src, dst, 1);
        predicted_dst = apply_homography(homography_matrix, src_points);
        num_inliers = compute_inliers(dst_points, predicted_dst, maxDistance);
        if (num_inliers > max_inliers)
            max_inliers = num_inliers;
            best_Homography = homography_matrix;
        end
    end
    disp(best_Homography);
    %using the best homography finding the inliers
    predicted_dst = apply_homography(best_Homography, src_points);
    distances = sqrt(sum((predicted_dst - dst_points).^2, 2));
    % find inliers based on distance threshold
    inliers_indices = distances < maxDistance;
    inlier_sets = matched_corner_sets_th(inliers_indices, :);
    src_points = [ inlier_sets(:, 1:2) ];
    dst_points = [ inlier_sets(:, 3:4) ];
    tform1 = compute_least_sq_homography(src_points, dst_points);
    tform = estgeotform2d(dst_points, src_points,'projective');
end
function [ncc_th, matched_corner_sets_th] = threshold_ncc(max_ncc_values, matched_corner_sets, threshold)
    ncc_th = [];
    matched_corner_sets_th = [];
    for i = 1 : length(max_ncc_values)
        if(max_ncc_values(i) >= threshold)
            %Appending if max_ncc_value is greater than threshold
            ncc_th(end+1) = max_ncc_values(i);
            matched_corner_sets_th(end+1, :) = matched_corner_sets(i, :);
        end
    end
end
function image_patch = create_image_patch(row_center, col_center, image, w_size)
    img_size = size(image);
   
    if(row_center - (w_size-1)/2 > 1)
        start_i = row_center - (w_size-1)/2;
    else
        start_i = 1;
    end
       
    if(col_center - (w_size-1)/2 > 1)
        start_j = col_center - (w_size-1)/2;
    else
        start_j= 1;
    end
    if(row_center + (w_size-1)/2 < img_size(1))
        end_i = row_center + (w_size-1)/2;
    else
        end_i = img_size(1);
    end
       
    if(col_center + (w_size-1)/2 < img_size(2))
        end_j = col_center + (w_size-1)/2;
    else
        end_j = img_size(2);
    end
    % Creating patches
    if (size(image, 3) == 3)
        gray_image = rgb2gray(image);
    else
        gray_image = image;
    end
%     image_patch = gray_image(start_i : end_i, start_j : end_j);
    image_patch = gray_image(start_i : end_i, start_j : end_j);
    required_patch_size = [w_size w_size];
    patch_size = size(image_patch);
    if (patch_size(1) ~= required_patch_size(1))
        r_size = w_size - patch_size(1);
        padSize = [r_size, 0];
        if (row_center - (w_size-1)/2 < 1)
            image_patch = padarray(image_patch, padSize, 0, 'pre');
        elseif (row_center + (w_size-1)/2 > img_size(1))
            image_patch = padarray(image_patch, padSize, 0, 'post');
        end
    end
    if (patch_size(2) ~= required_patch_size(2))
        c_size = w_size - patch_size(2);
        padSize = [0, c_size];
        if (col_center - (w_size-1)/2 < 1)
            image_patch = padarray(image_patch, padSize, 0, 'pre');
        elseif (col_center + (w_size-1)/2 > img_size(2))
            image_patch = padarray(image_patch, padSize, 0, 'post');
        end
    end
    image_patch = im2double(image_patch);
end
function [max_ncc_values, matched_corner_sets] = norm_cross_corr(row_col1, image1, row_col2, image2)
    w_size = 9;
    max_ncc_values = [];
    matched_corner_sets = [];
    img1_corner = zeros(1,2);
    img2_corner = zeros(1,2);
    for i = 1 : length(row_col1)
        row_center_img1 = row_col1(i, 1);
        col_center_img1 = row_col1(i, 2);
        image1_patch = create_image_patch(row_center_img1, col_center_img1, image1, w_size);
        mean_img1_patch = mean(image1_patch( : ));
        max_ncc = 0;
        for j = 1:length(row_col2)
            row_center_img2 = row_col2(j, 1);
            col_center_img2 = row_col2(j, 2);
            image2_patch = create_image_patch(row_center_img2, col_center_img2, image2, w_size);
            mean_img2_patch = mean(image2_patch( : ));
            product = (image1_patch - mean_img1_patch) .* (image2_patch - mean_img2_patch);
            numerator_term = sum( product( : ) );
    
            s1 = (image1_patch - mean_img1_patch) .^ 2;
            s1 = sum( s1( : ) );
            s2 = (image2_patch - mean_img2_patch) .^ 2;
            s2 = sum( s2( : ) );
            denominator_term = sqrt(s1 * s2);
            ncc = numerator_term / denominator_term;
            if (max_ncc < ncc)
                img1_corner = [row_center_img1, col_center_img1];
                img2_corner = [row_center_img2, col_center_img2];
                max_ncc = ncc;
            end
        end
        max_ncc_values(i) = max_ncc;
        matched_corner_sets(i, :) = [img1_corner, img2_corner]; 
    end
end
function [r, c] = harris_corner_alg(image)
    % checks if the image is coloured and converts to gray
    if (size(image, 3) == 3)
        gray_image = rgb2gray(image);
    else
        gray_image = image;
    end
   
    %Prewitt masks
    prewitt_X = [-1, -2, -1; 0, 0, 0; 1, 2, 1];
    prewitt_Y = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
   
    % Calculating Ix, Iy, Ix2, Iy2, Ix.*Iy
    Ix = conv2(gray_image, prewitt_X, 'same');
    Iy = conv2(gray_image, prewitt_Y, 'same');
   
    %The use of the Gaussian filter in Harris corner detector helps to improve
    % the accuracy and robustness by reducing the noise of the corner detection
    % algorithm.
    size_window = 3;
    gaussian_filter = gausswin(size_window.^2);
    Ix2 = conv2(Ix .^ 2, gaussian_filter, 'same');
    Iy2 = conv2(Iy .^ 2, gaussian_filter, 'same');
    Ixy = conv2(Ix .* Iy, gaussian_filter, 'same');
   
    %M = [ Ix2  Ixy ]
    %    [ Ixy  Iy2 ]
    %Calculating R value (det(M) - k * sq(trace(M)))
    det = (Ix2 .* Iy2) - (Ixy .* Ixy);
    trace = Ix2 + Iy2;
    k = 0.04;
    R = det - k*(trace .^ 2);
   
    % non-max suppression
    
    size_window = 11;
    threshold = 1000000;
    median_filtered = ordfilt2(R, size_window^2, ones(size_window));
   
    % threshold
    harris = (R == median_filtered) & (R> threshold);
   
    % Detect corners using the Harris corner detector
    % Uncomment the below line to use inbuild Harris Corner Detector Function
    %corners = detectHarrisFeatures(gray_image);
   
    % Finding the row and column number of the image where the harris value is
    % greater than or equal to one
    [r, c] = ind2sub(size(harris), find(harris >= 1));
   
    % Display the image with the detected corners
    imshowpair(image, harris, 'blend');
    title("original image and detected corners");
    waitforbuttonpress;
    close;
end
function gray_images = convert_grayScale(images)
    gray_images = [];
    for i = 1:size(images, 4)
        image = im2gray(images(:, :, :, i));
        gray_images = cat(4, gray_images, image);
    end
end

function display_images(images, image_Title)
    %This function displays the images
    for i = 1:size(images, 4)
        imshow(images(:, :, :, i));
        title(image_Title);
        pause(0.01);
    end
    waitforbuttonpress;
    close;
end
function images = read_images_from_path(path)
    % This function reads in images from a specified path and returns them as
    % an array of images.
   
    % List all files in the specified path
    file_list = dir(path);
    % Initialize an empty array to store the images
    images = [];
   
    % Loop through each file in the path
    for i = 1:length(file_list)
        % Get the current file name
        file_name = file_list(i).name;
   
        % Check if the current file is an image file
        if endsWith(file_name, {'.JPG', '.jpeg', '.png', '.bmp', '.gif'})
            % Read in the image and add it to the array of images
            image = imread(fullfile(path, file_name));
            images = cat(4, images, image);
        end
    end
    % Display a message if no images were found
    if isempty(images)
        disp('No images found in specified path.')
    end
end