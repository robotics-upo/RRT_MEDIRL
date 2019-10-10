function [path] = f_extract_path(map,label)
% function [path,goal_i,goal_j,path_img_norm] = f_extract_path(map,label)
n=2;
size_label = size(label);
size_i = size_label(1);
size_j = size_label(2);

path_img_norm=label>125;


% se=strel('sphere',n);
% path_img=imerode(label,se);
% path_img=imdilate(path_img,se);
% path_img=imerode(path_img,se);
% path_img_norm = path_img>125;
map_norm = double(map)/255;
% imshow(path_img)

goal_mix = find((map_norm > 0.5 & map_norm < 0.7)==1);
if isempty(goal_mix)
    goal_j = 1;
    goal_i = 1;
else
    goal_j = floor(goal_mix/size_j);
    goal_i = goal_mix-goal_j*size_j;
    
    goal_j = round(sum(goal_j)/length(goal_j));
    goal_i=round(sum(goal_i)/length(goal_i));
end


path_img_norm = -path_img_norm;
path_img_norm(100,100) = 1;
path_img_norm(goal_i,goal_j) = -1;

for k = 1:400
    for i = 2:199
        for j=2:199
            if path_img_norm(i,j)>0
                if path_img_norm(i+1,j) == -1
                    path_img_norm(i+1,j) = path_img_norm(i,j)+1;
                end
                if path_img_norm(i-1,j) == -1
                    path_img_norm(i-1,j) = path_img_norm(i,j)+1;
                end
                if path_img_norm(i,j+1) == -1
                    path_img_norm(i,j+1) = path_img_norm(i,j)+1;
                end
                if path_img_norm(i,j-1) == -1
                    path_img_norm(i,j-1) = path_img_norm(i,j)+1;
                end
                if path_img_norm(i+1,j-1) == -1
                    path_img_norm(i+1,j-1) = path_img_norm(i,j)+1;
                end
                if path_img_norm(i+1,j+1) == -1
                    path_img_norm(i+1,j+1) = path_img_norm(i,j)+1;
                end
                if path_img_norm(i-1,j+1) == -1
                    path_img_norm(i-1,j+1) = path_img_norm(i,j)+1;
                end
                if path_img_norm(i-1,j-1) == -1
                    path_img_norm(i-1,j-1) = path_img_norm(i,j)+1;
                end
            end
        end
    end
end
% imshow(path_img_norm/max(max(path_img_norm)))

path = [];
path_x = [];
path_y = [];
path_img_norm(goal_i,goal_j);
if path_img_norm(goal_i,goal_j) ~= -1
    cont = path_img_norm(goal_i,goal_j);
    i = goal_i;
    j = goal_j;
    path_x = [path_x i];
    path_y = [path_y j];
    while cont~=0
        if path_img_norm(i-1,j) == cont-1
            i = i-1;
        elseif path_img_norm(i+1,j) == cont-1
            i = i+1;
        elseif path_img_norm(i,j+1) == cont-1
            j = j+1;
        elseif path_img_norm(i,j-1) == cont-1
            j = j-1;
        elseif path_img_norm(i-1,j+1) == cont-1
            i = i-1; j = j+1;
        elseif path_img_norm(i-1,j-1) == cont-1
            i = i-1; j = j-1;
        elseif path_img_norm(i+1,j+1) == cont-1
            i = i+1; j = j+1;
        elseif path_img_norm(i+1,j-1) == cont-1
            i = i+1; j = j-1;
        end
        path_x = [path_x i];
        path_y = [path_y j];
        cont = cont-1;
    end
    path = [path_x(end:-1:1); path_y(end:-1:1)];
end


end

