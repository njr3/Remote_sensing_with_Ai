clear;close all;clc;

%% 1. ������ ����� ��η� ���� �� �ڷ� �ҷ�����
cd ..\data
% ���� �ҷ�����
raw_cali_dat = readtable('classification_tr.csv');
raw_vali_dat = readtable('classification_va.csv');
% ��������
fselect_cali_dat = raw_cali_dat(:,{'spr_b3', 'win_b3', 'class'});
fselect_vali_dat = raw_vali_dat(:,{'spr_b3', 'win_b3', 'class'});
% ���� �۾��� ���� �� Ŭ������ ����
cali_dat = fselect_cali_dat(133:493,:);
vali_dat = fselect_vali_dat(33:121,:);

%% 2. Normalize
% Min�� Max ���� ��, cali vali ���ϰ� ������ �°� ����
dummy1 = [cali_dat; vali_dat];
min_arr = min(table2array(dummy1))
max_arr = max(table2array(dummy1))
c_min_mat = repmat(min_arr, size(cali_dat,1),1);
v_min_mat = repmat(min_arr, size(vali_dat,1),1);
c_max_mat = repmat(max_arr, size(cali_dat,1),1);
v_max_mat = repmat(max_arr, size(vali_dat,1),1);
% ������ ���� cali vali ������ array�� ����
t_cali = table2array(cali_dat);
t_vali = table2array(vali_dat);
% normalize ����. �� ��, target������ ǥ��ȭ���� ������.
inst_ncali_dat = (t_cali - c_min_mat)./(c_max_mat - c_min_mat);
inst_nvali_dat = (t_vali - v_min_mat)./(v_max_mat - v_min_mat);
inst_ncali_dat(:,end) = t_cali(:,end);
inst_nvali_dat(:,end) = t_vali(:,end);
% normalize �� ����� table �������� ����
ncali_dat = array2table(inst_ncali_dat, 'VariableNames', cali_dat.Properties.VariableNames);
nvali_dat = array2table(inst_nvali_dat, 'VariableNames', vali_dat.Properties.VariableNames);


%% 3. kNN �з� - k=1�� ����
train_x = ncali_dat(:,1:end-1);
train_y = ncali_dat(:,end);
test_x = nvali_dat(:,1:end-1);
test_y = nvali_dat(:,end);

% fit a kNN classification model
mdl = fitcknn(train_x, train_y, 'NumNeighbors', 1);
% 'NumNeighbors' - Number of nearest neighbors to find
% 'Exponent' - Minkowski distance exponent. 2 (default)
% 'Standardize' - Flag to standardize predictors. false (default)
% 'OptimizeHyperparameters' - Parameters to optimize. 
%       'none' (default), 'auto' (Use {'Distance','NumNeighbors'}), 'all'

% apply the model to test data and compute test error
pred = predict(mdl, test_x);
cros_t = crosstab(table2array(test_y), pred)
OA = sum(table2array(test_y) == pred)/numel(test_y)*100


%% 4. kNN �з� - ������ k ã��
accum_OA=[];
k_range = 1:20;
for kk = k_range
    mdl = fitcknn(train_x, train_y, 'NumNeighbors', kk);
    pred = predict(mdl, test_x);
    OA = sum(table2array(test_y) == pred)/numel(test_y)*100;
    accum_OA = [accum_OA; OA];
end

figure;plot(k_range, accum_OA, '-o')
title('OA trend by the k increase');
xlabel('k');
ylabel('OA (%)');

best_k = k_range(find(accum_OA == max(accum_OA)));
disp(['The best k is ', num2str(best_k)])


%% 5. kNN �з� - ������ k �� �з���

mdl = fitcknn(train_x, train_y, 'NumNeighbors', best_k);
pred = predict(mdl, test_x);
cros_t = crosstab(table2array(test_y), pred)
OA = sum(table2array(test_y) == pred)/numel(test_y)*100




