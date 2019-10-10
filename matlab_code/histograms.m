
dir_dataset = 'workspace'


nbars = 4; % FCN, RTIRL, RLT
sets = 3;  

for trial = 1:3
    t = 1;
    for i=0:10
        for j=0:250
            try
                dist_set(trial,t,j+1) = load(sprintf(strcat(dir_dataset,'/dist_set%i/file_%i_%i.csv'), trial,i,j));
            end
        end
        t=t+1;
    end
    med_met(trial) = sum(sum(dist_set(trial,:,:)))/numel(dist_set(1,:,:));
    
    med_met_dev(trial) = sqrt(sum(sum((med_met(trial) - dist_set(trial,:,:)).^2))/numel(dist_set(trial,:,:)));
    med_met_err(trial) = med_met_dev(trial) / sqrt(numel(dist_set(trial,:,:)));
end


%               set 1     set 2     set 3
% FCN +++++++++++++++++++++++++++++++++++++

%-----MET----------------------------------
fcn_met =      [0.3737,   0.3275,   0.4102];
fcn_met_dev =  [0.3954,   0.3826,   0.3643];
fcn_met_err =  [0.0323,   0.0312,   0.0297];

% RLT +++++++++++++++++++++++++++++++++++++

%-----MET----------------------------------
rlt_met =      [0.3560,   0.3581,   0.3936];
rlt_met_dev =  [0.3784,   0.3444,   0.4144];
rlt_met_err =  [0.0266,   0.0218,   0.0262];

% RTIRL +++++++++++++++++++++++++++++++++++-

%-----MET----------------------------------
rt_met =       [0.3985,   0.4346,   0.3999];
rt_met_dev =   [0.4315,   0.3826,   0.4512];
rt_met_err =   [0.0273,   0.0242,   0.0285];





d_err = zeros(sets,nbars);
met = zeros(sets,nbars);
m_err = zeros(sets,nbars);

for i=1:sets
  
        met(i,:) = [fcn_met(1,i) rlt_met(1,i)  rt_met(1,i) med_met(1,i)]; % normrrt_dissimilarity(1,i)
        m_err(i,:) = [fcn_met_err(1,i) rlt_met_err(1,i) rt_met_err(1,i) med_met_err(1,i)]; 
end




figure(1)
h = bar(met);
set(h,'BarWidth',1);    % The bars will now touch each other
set(gca,'YGrid','on');
set(get(gca,'YLabel'),'String','Distance metric (m)');
xlabel('Sets');
legend('FCN-RRT*', 'RLT', 'RTIRL','RRT-MEDIRL');
title('Distance metric');
xlim([0 (sets+1)]);
hold on;
numgroups = size(met, 1); 
numbars = size(met, 2);
groupwidth = min(0.8, numbars/(numbars+1.5));

for i = 1:numbars
    % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
    % Aligning error bar with individual bar
    x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);
    errorbar(x, met(:,i), m_err(:,i), 'k', 'linestyle', 'none');
end
hold off;

legend('FCN-RRT*', 'RLT', 'RTIRL','RRT-MEDIRL');

ylim([0,0.6])


