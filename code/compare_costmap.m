
dir_dataset = 'workspace'

for i=0:250
    try
        costmap(i+1,:,:) = load(sprintf( strcat(dir_dataset,'/csv_files/file_%i.csv'), i));
    end
end

for i=0:250
    try
        costmap_label_aux(:,:) = load(sprintf( strcat(dir_dataset,'/csv_files_label/file_%i.csv'), i));
        costmap_label(i+1,:,:) = costmap_label_aux(:,1:(end-1));
    end
end

for i=0:250
    try
        labels(i+1,:,:) = load(sprintf( strcat(dir_dataset,'/labels/file_%i.csv'), i));
    end
end

for i=0:250
    try
        rrt_out(i+1,:,:) = load(sprintf( strcat(dir_dataset,'/rrt_out/file_%i.csv'), i));
    end
end

for i=0:250
    try
        map_img(i+1,:,:) = load(sprintf( strcat(dir_dataset,'/map/F_%i.csv'), i));
    end
end

for j=1:size(map_img,1)
    p_label{j} = f_extract_path(squeeze(map_img(j,:,:))*255.0,squeeze(labels(j,:,:))*255.0);
    p_out{j} = f_extract_path(squeeze(map_img(j,:,:))*255.0,squeeze(rrt_out(j,:,:))*255.0);
end

t=0;

for j=1:size(map_img,1)
    
    A=zeros(200);
    cost_label = 0;
    cost_out = 0;
    for k=1:(size(p_label{j},2)-1)
        cost_label = cost_label+ sqrt((p_label{j}(1,k)-p_label{j}(1,k+1)).^2+(p_label{j}(2,k)-p_label{j}(2,k+1)).^2)*(costmap_label(p_label{j}(1,k),p_label{j}(2,k))+costmap_label(p_label{j}(1,k+1),p_label{j}(2,k+1)))/2;
        A((p_label{j}(1,k)),(p_label{j}(2,k))) = 255;
    end
    for k=1:(size(p_out{j},2)-1)
        cost_out = cost_out+ sqrt((p_out{j}(1,k)-p_out{j}(1,k+1)).^2+(p_out{j}(2,k)-p_out{j}(2,k+1)).^2)*(costmap_label(p_out{j}(1,k),p_out{j}(2,k))+costmap_label(p_out{j}(1,k+1),p_out{j}(2,k+1)))/2;
    end
    cost_label = cost_label/(size(p_label{j},2)-1);
    cost_out = cost_out/(size(p_out{j},2)-1);
    
    if isfinite((cost_label-cost_out)/cost_label)
        error_i(j) = (cost_out -cost_label)/cost_label;
        t=t+1;
    else
        error_i(j) = 100;
    end
end


error_i = error_i(error_i~=100);

error_med = sum(error_i)/length(error_i)

error_met_dev = sqrt(sum((error_med - error_i).^2)/length(error_i));
error_met_err = error_met_dev / sqrt(length(error_i))


h(1) = cdfplot(error_i);
xlabel('Relative error of the path','FontSize', 16);
t = 'Relative error of the ground-truth cost of the paths';
title(t, 'FontSize', 16);
set(h(1),'LineWidth', 2.75);



i = 1; %image plotted

figure(6)
subplot(2,3,[1,4]);
imshow(squeeze(map_img(i,:,:)),'Border','tight');
title('Map');
subplot(2,3,2);
imshow(squeeze(costmap_label(i,:,:)),'Border','tight')
title('Costmap Groundtruth');
subplot(2,3,5);
imshow(squeeze(costmap(i,:,:)*5),'Border','tight')
title('Computed Costmap');
subplot(2,3,3);
imshow(squeeze(labels(i,:,:)),'Border','tight')
title('Label');
subplot(2,3,6);
imshow(squeeze(rrt_out(i,:,:)),'Border','tight')
title('RRT-MEDIRL output');


i=1
bright = 3;
figure(1);
A=zeros(200,200,3);
A_1 = squeeze(costmap_label(i,:,:)*bright/2);
A_1(map_img(i,:,:)>0.1) = squeeze(map_img(i,map_img(i,:,:)>0.1)*bright);
A_1(labels(i,:,:)>0.1) = squeeze(labels(i,labels(i,:,:)>0.5));
A(:,:,1) = A_1;
A(:,:,2) = squeeze(map_img(i,:,:)*bright);
A(:,:,3) = squeeze(map_img(i,:,:)*bright)+squeeze(labels(i,:,:)) ;
imshow(A);

figure(2);
B=zeros(200,200,3);
B_1 = squeeze(costmap(i,:,:)*bright);
B_1(map_img(i,:,:)>0.1) = squeeze(map_img(i,map_img(i,:,:)>0.1)*bright);
B(:,:,2) = B_1;
B(:,:,1) = squeeze(map_img(i,:,:)*bright)+squeeze(rrt_out(i,:,:)) ;
B(:,:,3) = squeeze(map_img(i,:,:)*bright);
c_costmap = zeros(200,200,3);
c_costmap(:,:,2) = squeeze(costmap(i,:,:)*bright);
%
imshow(B);







