clear all
close all
clc
filename_patch = './temp_files/patches_heatmap_test.txt';
filename_score = './temp_files/MTF_score_heatmap_test.txt';
img = imread('./temp_files/heatmap_test_gray.jpg');
map = zeros(size(img));

f= fopen(filename_patch);
if(fgetl(f) == -1)
    M = [];% file is empty
else
    M = csvread(filename_patch);
end
fclose(f);

fileID = fopen(filename_score,'r');
score = fscanf(fileID,'%f');
fclose(fileID);

size(M)
size(score)
for j = 1:size(M,1)
    midr = M(j,1);
    midc = M(j,2);
    map(midr,midc) = score(j);
end
[n,m] = size(map);
bdry_mrg = 24;
map = map(bdry_mrg:n-bdry_mrg,bdry_mrg:m-bdry_mrg);
%map = imagesc(map)
img = img(bdry_mrg:n-bdry_mrg,bdry_mrg:m-bdry_mrg);
img = double(img).*map;
imagesc(map)
figure,imshow(img,[])
imwrite(img,'heatmap_test.png')
figure, imagesc(img)
