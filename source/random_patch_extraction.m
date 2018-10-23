clear all;
close all;
clc
patch_size = 48;
num_patches = 300;
bdry_mrg = patch_size/2;

path_roi_images = './Images/';
path_patches ='./patches/';
total_images = 10

for ind=1:total_images
    ind
    img = imread(strcat(path_roi_images,int2str(ind),'.jpg'));
    [n,m] = size(img);
	rnd_x = randi([bdry_mrg+1, m-bdry_mrg],1,num_patches);
	rnd_y = randi([bdry_mrg+1, n-bdry_mrg],1,num_patches);
                    
	fileID_h = fopen(strcat(path_patches,int2str(ind),'.txt'),'w');
    first_patch = 0;
    
	for i=1:num_patches
            if(first_patch == 0)
               fprintf(fileID_h,'%d,%d',[rnd_y(i) rnd_x(i)]);
               first_patch = 1;
            else
               fprintf(fileID_h,'\n%d,%d',[rnd_y(i) rnd_x(i)]);
            end
    end
    
    fclose(fileID_h);
end
