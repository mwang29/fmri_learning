demographics_Data_EPI3 = xlsread('demographics.xlsx');
FHA_EPI3_binary_extraction = sum(demographics_Data_EPI3(:,[9,10]),2);
FHA_EPI3_binary = (FHA_EPI3_binary_extraction==0); %1 = FHA negative (no first or second degree relative)
FHA_neg_indices = find(FHA_EPI3_binary);
FHA_pos_indices = find(~FHA_EPI3_binary);
for i = 1:length(FHA_neg_indices)
    neg_indices(i) = FHA_neg_indices(i);
end

for i = 1:length(FHA_pos_indices)
    pos_indices(i) = FHA_pos_indices(i);
end


save('alcohol.mat', 'neg_indices', 'pos_indices')