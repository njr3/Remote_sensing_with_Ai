% 'BoxConstraint'(������������) : ������ �����ϴ� �������� ����Ǵ� �ִ� ������ �����ϰ� ���������� ����(����ȭ)�ϴ�
%                               �� ������ �Ǵ� ����� ���� ���� ������ �ø��� SVM �з��Ⱑ �� ���� ����Ʈ ���͸� �Ҵ��մϴ�. 
%                               �׷���, ���� ���� ������ �ø��� �Ʒ� �ð��� �� ����� �� �ֽ��ϴ�.(����Ʈ
%                               : 1)
% 'KernelFunction'(Ŀ���Լ�) : 'KernelFunction'�� �Բ� Ŀ�� �Լ� �̸��� ��ǥ�� ���еǾ� �����˴ϴ�.
%                              'gaussian' �Ǵ� 'rbf' - ���콺 Ŀ�� �Ǵ� ��� ���� �Լ�(RBF) Ŀ�η�, ���� Ŭ���� �н��� ���� ����Ʈ ����
%                              'linear' -  ���� Ŀ�η�, 2-Ŭ���� �н��� ���� ����Ʈ ���� 
%                              'polynomial' - ���׽� Ŀ��. 'PolynomialOrder',q�� ����Ͽ� ������ q�� ���׽� Ŀ���� ����%
% 'PolynomialOrder' : ���׽� Ŀ�� �Լ� ������, 'PolynomialOrder'�� �Բ� ���� ������ ��ǥ�� ���еǾ�
%                     �����˴ϴ�.(����Ʈ : 3)
% 'KernelScale'(Ŀ�� ������) : 'KernelScale'�� �Բ� 'auto' �Ǵ� ���� ��Į�� ��ǥ�� ���еǾ� �����˴ϴ�. 
%                              ����Ʈ����� ���� ���� ��� X�� ��� ��Ҹ� KernelScale�� ������ �����ϴ�. 
%                              �׷� ����, ������ Ŀ�� �븧(Norm)�� �����Ͽ� �׶� �����
%                              ����մϴ�.(����Ʈ :1)
% 'Standardize'(���� ���� �����͸� ǥ��ȭ) : ���� ���� �����͸� ǥ��ȭ�ϴ� �÷��׷�, 'Standardize'�� �Բ� true(1) �Ǵ� false(0)�� ��ǥ�� ���еǾ� �����˴ϴ�.

% 'OptimizeHyperparameters'(����ȭ�� ���) : 'none'(����ȭ����)/
%                                           'auto'(BoxConstraint,KernelScale �� ����)
%                                           'all'(���)
% BoxConstraint : fitcsvm�� �⺻������ ���� [1e-3,1e3]���� �α� �����ϸ��� ��� �� �߿��� �˻��� �����մϴ�.
% KernelScale : fitcsvm�� �⺻������ ���� [1e-3,1e3]���� �α� �����ϸ��� ��� �� �߿��� �˻��� �����մϴ�.
% KernelFunction : fitcsvm�� 'gaussian', 'linear', 'polynomial' �߿��� �˻��� �����մϴ�.
% PolynomialOrder : fitcsvm�� ���� [2,4] �� ���� �߿��� �˻��� �����մϴ�.
% Standardize : fitcsvm�� 'true' �� 'false' �߿��� �˻��� �����մϴ�.
%
% 'IterationLimit'(��ġ����ȭ �ݺ��� �ִ� Ƚ��) : ��ġ ����ȭ �ݺ��� �ִ� Ƚ����, 'IterationLimit'�� �Բ� ���� ������ ��ǥ�� ���еǾ� �����˴ϴ�.
%                                               ����ȭ ��ƾ�� ���������� ���ŵǴ��� ���ο� ������� ����Ʈ��� �Ʒõ� ���� ��ȯ�մϴ�.
%                                               (����Ʈ : 1e8)
% 'HyperparameterOptimizationOptions' : ����)https://kr.mathworks.com/help/stats/fitcsvm.html?s_tid=doc_ta#bt9w6j6-OptimizeHyperparameters
% Grid search ���� �ð� ���� �ɸ��� �� ����!!!
%% classification
clc;clear;
path = 'D:\class_SVM\matlab\';

% open calibration file as table
cali = readtable('climate_zone_cali.csv');
cali_input = table2array(cali(:,1:size(cali,2)-1)); % calibration input
cali_target = table2array(cali(:,12)); %calibration target

% open validation file as table
vali = readtable'climate_zone_vali.csv']);
vali_input = table2array(vali(:,1:size(cali,2)-1)); % validation input
vali_target = table2array(vali(:,12)); %validation target

classes = unique(cali_target); % class type
SVMModels = cell(size(classes,1),1); 

for j = 1:numel(classes);
    % Create binary classes for each classifier
    indx = strcmp(cali_target,classes(j)); 
    % Train an SVM classifier using cali_input data and indx and Store the classifier in a cell of a cell array.
%     SVMModels{j} = fitcsvm(cali_input,indx,'ClassNames',[false true],'Standardize',true,...
%          'KernelFunction','rbf','BoxConstraint',1); 
%    optimization   
   SVMModels{j} = fitcsvm(cali_input,indx,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Optimizer', 'gridsearch','MaxObjectiveEvaluations',5),'IterationLimit',5)
end

for j = 1:numel(classes);
    [~,score] = predict(SVMModels{j},vali_input);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end
[~,maxScore] = max(Scores,[],2);

% if you want to get validation results as label name,
% for i = 1:size(maxScore,1); 
%     output{i,1} = classes{maxScore(i,1),1};
% end


%% regression
clc;clear;
path = 'G:\class_SVM\matlab\';
cali = readtable([path 'PM10_cal_AIRS.csv']); % open calibration file
vali = readtable([path 'PM10_val_AIRS.csv']); %v open validation file

SVRmodel = fitrsvm(cali,'PM10','KernelFunction','gaussian','KernelScale','auto','Standardize',true);
% optimization
%SVRmodel = fitrsvm(cali,'PM10','OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('Optimizer', 'gridsearch'));
%'IterationLimit',5

% sturct ���� �ȿ��� parameter ����?

test = predict(SVRmodel,vali);
scatter(table2array(vali(:,11)),test)

