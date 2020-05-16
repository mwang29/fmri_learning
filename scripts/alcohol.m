%% FHA matlab code to differentiate FHA positive and negative patients
demographics_Data_EPI3 = xlsread('../data/demographics.xlsx');
FHA_EPI3_binary_extraction = sum(demographics_Data_EPI3(:,[9,10]),2);
FHA_EPI3_binary = (FHA_EPI3_binary_extraction==0); %1 = FHA negative (no first or second degree relative)
neg_indices = find(FHA_EPI3_binary);
pos_indices = find(~FHA_EPI3_binary);
clearvars FHA_EPI3_binary FHA_EPI3_binary_extraction demographics_Data_EPI3
save('../data/alcohol.mat')