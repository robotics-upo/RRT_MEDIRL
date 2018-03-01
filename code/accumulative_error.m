clear,close all,clc


dir_dataset = 'workspace'


trial = 3;

t = 1;
route_rtirl = sprintf( strcat(dir_dataset,'/upo_nav_irl/results/RTIRL/set%i/Metrics.txt'), trial);
rtirl_data = load(route_rtirl);

route_rlt = sprintf( strcat(dir_dataset,'/upo_nav_irl/results/RLT/set%i/Metrics.txt'), trial);
rlt_data = load(route_rlt);

route_fcn = sprintf( strcat(dir_dataset,'/upo_nav_irl/results/FCN/set%i/Metrics.txt'), trial);
fcn_data = load(route_fcn);
%[1,3,4,7,8] %set2

for i=0:10
    for j=0:250
        try
            mu_medirl(t,j+1) = load(sprintf( strcat(dir_dataset,'/dist_set%i/file_%i_%i.csv'), trial,i,j));
        end 
    end
    t=t+1;
end



figure(1)
% Distance metric
mu_rtirl = rtirl_data(:,3);
mu_rlt = rlt_data(:,3);
mu_fcn = fcn_data(:,3);
h(1) = cdfplot(mu_fcn);
hold on;
h(2) = cdfplot(mu_rlt);
h(3) = cdfplot(mu_rtirl);
h(4) = cdfplot(mu_medirl(:));
hold off;

legend([h(1), h(2), h(3),h(4)], {'FCN-RRT*', 'RLT', 'RTIRL','RRT-MEDIRL'}, 'FontSize', 18);
xlabel('Distance metric u (m)','FontSize', 16);
t = sprintf('Empirical CDF - SET %i', trial);
title(t, 'FontSize', 16); 

set(h(1), 'LineStyle', '--', 'LineWidth', 2.75, 'Color', 'b');
set(h(2), 'LineStyle', '-', 'LineWidth', 2.75, 'Color', 'g');
set(h(3), 'LineStyle', '-.', 'LineWidth', 2.75, 'Color', 'r');
set(h(4), 'LineStyle', '-.', 'LineWidth', 2.75, 'Color', 'm');

