% Load the two images that you want to use
path = "C:\Users\Shiva Kumar Dande\Desktop\Semester 2\CV_Project\Project 2_Image_mosaicing\DanaOffice\DanaOffice";
input_images = read_images_from_path(path);
img1 = input_images(:, :, :, 1);
img2 = input_images(:, :, :, 2);

% Display the images
figure(1);
imshow(img1);
figure(2);
imshow(img2);

% Use Matlab's ginput function to collect mouse clicks for the corners
% of the image you want to insert into the frame and the corresponding
% points in the second view for the corners of the rectangle into which
% the first image should be warped.

% Collect points from image1
figure(1);
disp('Select four points in image1 (in order: top-left, top-right, bottom-right, bottom-left)');
[x1, y1] = ginput(4);

% Collect points from image2
figure(2);
disp('Select four points in image2 (in order: top-left, top-right, bottom-right, bottom-left)');
[x2, y2] = ginput(4);

% Calculate the homography matrix that maps the corners of the first
% image to the corresponding points in the second image using the
% clicked points.
H = homography(x1, y1, x2, y2);

% Use the homography matrix to warp the first image into the rectangle in
% the second image.
[img1_warped, x_min, y_min, x_max, y_max] = warp_image(img1, H);

% Display the resulting image
figure(3);
imshow(img1_warped);

% Save the resulting image
imwrite(img1_warped, 'result.jpg');

% % Define the homography function
% function H = homography(x1, y1, x2, y2)
% A = zeros(8, 9);
% for i = 1:4
% A(2*i-1,:) = [-x1(i) -y1(i) -1 0 0 0 x1(i)*x2(i) y1(i) x2(i) x2(i)];
% A(2i,:) = [0 0 0 -x1(i) -y1(i) -1 x1(i)*y2(i) y1(i)*y2(i) y2(i)];
% end
% [~, ~, V] = svd(A);
% H = reshape(V(:,end), [3,3])';
% end

function H = homography(x1, y1, x2, y2)
    % construct matrix A
    A = zeros(2*size(x1,1),9);
    for i = 1:size(x1,1)
        A(2*i-1,:) = [-x1(i) -y1(i) -1 0 0 0 x1(i)*x2(i) y1(i)*x2(i) x2(i)];
        A(2*i,:) = [0 0 0 -x1(i) -y1(i) -1 x1(i)*y2(i) y1(i)*y2(i) y2(i)];
    end
    
    % solve for H using SVD
    [~,~,V] = svd(A);
    H = reshape(V(:,end),3,3)';
end


% Define the warp_image function
function [img_warped, x_min, y_min, x_max, y_max] = warp_image(img, H)
% Find the corners of the image after warping
[rows, cols, ~] = size(img);
corners = [1, 1, cols, cols; 1, rows, rows, 1; 1, 1, 1, 1];
warped_corners = H * corners;
warped_corners = warped_corners ./ warped_corners(3,:);
% Find the minimum and maximum x and y coordinates after warping
x_min = floor(min(warped_corners(1,:)));
y_min = floor(min(warped_corners(2,:)));
x_max = ceil(max(warped_corners(1,:)));
y_max = ceil(max(warped_corners(2,:)));

% Create a meshgrid of the warped image coordinates
[X, Y] = meshgrid(x_min:x_max, y_min:y_max);
coords = [X(:), Y(:), ones(length(X(:)),1)]';

% Invert the homography matrix
H_inv = inv(H);

% Map the warped coordinates back to the original image
warped_coords = H_inv * coords;
warped_coords = warped_coords ./ warped_coords(3,:);
warped_x = reshape(warped_coords(1,:), size(X));
warped_y = reshape(warped_coords(2,:), size(Y));

% Interpolate the values of the original image at the warped coordinates
img_warped = zeros(size(X,1), size(X,2), size(img,3));
for i = 1:size(img,3)
    img_warped(:,:,i) = interp2(double(img(:,:,i)), warped_x, warped_y, 'linear', 0);
end
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