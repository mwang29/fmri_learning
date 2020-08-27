# Optimizing predictive power of functional connectivity data with tangent spaces and convolutional neural networks
The aim of this research is to experiment with tangent space projection to increase predictive power and identifiability of functional connectivity data. 

To setup a conda virtual environment from the requirements.txt file, use
$ conda create --name <env> --file requirements.txt

The folder structure is as follows:

Thesis presentation and PDF
Overleaf document: https://www.overleaf.com/read/ntvwqwxszskz

Data - Includes original FCs from Glasser and Schaefer parcellations. Also includes .pickle files of tangent projected FCs to avoid performing tangent space projection repetitively.

Scripts - Main jupyter notebook files of task, twin, and subject identification. Also includes CNN demo for task identification and some other .py scripts.

Results - Visualization notebooks with Manuscript_figures and Data_Visualizations. Outputs are stored in subfolders corresponding to task, subject, or twins. 

Email michael20995@gmail.com with any questions. Thanks.

