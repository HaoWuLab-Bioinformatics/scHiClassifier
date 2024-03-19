## Overview
The folder "**4DN_Data**" contains the raw data 
The folder "**4DN_features**" contains the data of the enhancer and promoter, containing the sequences of the independent test sets and training sets. The first half of each file is labeled as 0, and the second half is labeled as 1.  
The folder "**Feature_extraction**" contains the data of the enhancer created by Liu et al.[1], containing the sequences of the independent test sets and training sets.   
The folder "**train&test**" contains the trained models on eight cell lines and the pre-trained models are trained on all cell lines for use or validation.  
The folder "**train&val**" contains the pre-trained DNA vectors provided in dna2vec[2].  
The file "**combo_hg19.genomesize**" is the code of the network architecture.  
The file "**dockerfile**" is the text file used to build Docker container images.  
 
## Usage 
At each step, we have given the execution code of NHLF cell line as an example, and users will get the test results of Enhancer-MDLF on NHLF after executing all of them according to the example.
### Step 0. Prepare dataset
We have provided enhancer training and test set data and labels for eight cell lines in the following directory:  
training set data : 'data/train/${cell line name}.fasta'  (**e.g.** 'data/train/NHLF.fasta')  
training set label : 'data/train/${cell line name}_y_train.txt'  (**e.g.** 'data/train/NHLF_y_train.txt')  
test set data : 'data/test/${cell line name}.fasta'  (**e.g.** 'data/test/NHLF.fasta')  
test set label : 'data/test/${cell line name}_y_test.txt'  (**e.g.** 'data/test/NHLF_y_test.txt')  
If users want to run Enhancer-MDLF using their own dataset , please organize the data in the format described above. 
### Step 1. Setup environment
First, in order to avoid conflicts between the project's packages and the user's commonly used environment, we recommend that users create a new conda virtual environment named test through the following script.  
`conda create -n test python=3.8`  
`conda activate test`  
Later, users can install all dependencies of the project by running the script:  
`pip install -r requirements.txt`  
### Step 2. Extract features of enhancers
Before running Enhancer-MDLF,users should extract features of enhancers through run the script to extract dna2vec-based features and motif-based features as follows:  
#### necessary input  
input = 'the data file from which you want to extract features.The file naming format is the same as in step 0.'  
cell_line = 'the cell line name for feature exrtraction'  
set = 'the extracted data for training or testing'  
#### run the script
(1) extract dna2vec feature  
`python dna2vec_code.py --input_file ${input} --cell_line ${cell_line} --set ${set}`   
**e.g.**`python dna2vec_code.py --input_file data/train/NHLF.fasta --cell_line NHLF --set train`  
**e.g.**`python dna2vec_code.py --input_file data/test/NHLF.fasta --cell_line NHLF --set test`  
(2) extract motif feature  
`python motif_find.py --input_file ${input} --cell_line ${cell_line} --set ${set}`  
**e.g.**`python motif_find.py --input_file data/train/NHLF.fasta --cell_line NHLF --set train`  
**e.g.**`python motif_find.py --input_file data/test/NHLF.fasta --cell_line NHLF --set test`  
The output feature files will be saved in the 'feature' directory
### Step 3. Run Enhancer-MDLF:  
Users can run the script as follows to compile and run Enhancer-MDLF:    
#### necessary input  
cell_line = 'the cell line name for train and prediction'  
#### run the script
`python main.py --cell_line ${cell_line}`    
e.g.`python main.py --cell_line NHLF`   
