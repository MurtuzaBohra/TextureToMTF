clear all;
close all;
clc
patch_size = 48;
bdry_mrg = patch_size/2;

img = imread('./temp_files/test.jpg');
[n,m,ignore] = size(img);
fileID_h = fopen('patches_test.txt','w');
first_patch = 0;

for i=bdry_mrg:n-bdry_mrg    
	for j=bdry_mrg:m-bdry_mrg
        if(first_patch == 0)
           fprintf(fileID_h,'%d,%d',[i j]);
           first_patch = 1;
        else
           fprintf(fileID_h,'\n%d,%d',[i j]);
        end
    end
end
fclose(fileID_h);
