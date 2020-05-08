import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["axes.labelsize"] = 14
baseline = pd.read_csv('HCP100_base_E200_LR0.001_R1_S0_Y1_20.csv')
tan_euclidean = pd.read_csv('HCP100_Taneuclid_E200_LR0.001_R1_S0_Y1_20.csv')
pca_recon = pd.read_csv('HCP100_recon_0.025_E200_LR0.001_R0_S0_Y1.csv')
df = pd.concat([baseline['Accuracy'], pca_recon['Accuracy'],
                tan_euclidean['Accuracy']], axis=1,
               keys=['Baseline', 'PCA Recon: 2.5% PCs', 'Tan Euclidean'])
df = df.melt(var_name='Pipelines', value_name='Test accuracy')
ax = sns.violinplot(x="Pipelines", y="Test accuracy", data=df).set_title(
    'HCP 8 Task Classification Accuracy of DL Piplines')
fig = ax.get_figure()
fig.savefig('accuracies.png')
# sns.set(style="whitegrid")
# tips = sns.load_dataset("tips")
# print(tips.head())
# ax = sns.violinplot(x="day", y="total_bill", data=tips)
