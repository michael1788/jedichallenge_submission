# JEDI

Supporting code for the JEDI GrandChallenge submission of team Tano.

## Challenge description
[The Jedi GrandChallenge description](https://www.covid19.jedi.group/): "Improve (in silico or others) methods to identify compounds with blocking interactions relevant to any SARS-CoV-2 target, by optimizing/accelerating the use of HPC (High Performance Computing), Artificial Intelligence, and provide experimental validation."    
More specifically: "The aim is to get high-quality lead compounds for multiple SARS-CoV-2 targets by cross-checking multiple simulation approaches. The top-scoring compounds will be analyzed experimentally using direct affinity assays, as well as viral assays on live SARS-CoV-2."    

**Timeline**:
- Start date (stage 1): May 4 2020
- Deadline (stage 1): ~~June 4 2020~~ (extended to 17 July 18:00 CET)   

## Description and motivation for our approach

## How to

**1.** Extract data from BindingDB    
After downloading the data from [here](https://www.bindingdb.org/bind/index.jsp) as a .csv file:
```
cd experiments/
python extract_data_from_BindingDB.py -m testing -aa 100 -an 200 -mp 2000 -ms 100 -d your_datapath/ -s your_savepath/ -kd -v  
```
This will do a test run (-m testing), consider a molecule active if the activity is <=100 nm (-aa 100) and inactive if the activity >200nm (-an 2000). The maximum protein length considered will be 2000 (-mp 2000), while the maximum SMILES length is 100 (-ms 100). It will only consider Kd value (-kd), and show print messages (-v). The BindingDB database saved as a .csv has to be in *your_datapath/*; your output file will be saved in *your_savepath/*. To do a full run, use *-m production*.     
If you set to True more than the kd values (you can add: -ki, -ic50, -ec50) and more than one measurement type was recorded for a given protein-molecule pair, they will be prioritized given the following order: kd > ki > IC50 > EC50. If a given pair of molecule and protein has more than one binary activity, it will be kept only if the activities are the same (e.g. all actives, that is <=100nm). If that condition it not met, the pair is fully removed from the dataset.

**2.** Extraction data representation   
```
cd experiments/
python get_data_representation.py -m testing -r morgan -d your_datapath/ -s your_savepath/ -n 3 -v
```
This will execute a test run (-m testing), extract the Morgan fingerprints (-r morgan) from the data in *your_datapath/*. The output will be saved in *your_savepath/*. Three workers (-n 3) will be use in parallel to speed up the run and print messages will be displayed (-v).   
The -r argument can be: protein &rarr; BERT latent representation, clm &rarr; CLM latent representation, morgan &rarr; Morgan fingerprint (radius=2, nBits=1024), binary &rarr; extract a dict for the data generator data id to label, maccs &rarr; MACCS keys. To do a full run, use -m production.

**3.** Cluster the protein sequences
First, get the data ready to be used by MMseqs2:
```
cd experiments/
python do_process_data_for_MMseqs2.py -d your_datapath/ -s your_savepath/ -v
```
Then, run MMseqs2 in the directory where you saved your data:
```
cd your_datapath/
mmseqs easy-cluster data.fasta clusterRes tmp --min-seq-id 0.8
```
This command will use a sequence identity of 0.8 to cluster the proteins.   
Finally, process the output from MMseqs2 for the cross-validation:
```
cd experiments/
python do_process_from_MMseqs2.py -d your_datapath/ -s your_savepath/ -v
```

**4.** Do the cross-validation folds based on the protein clusters
```
cd experiments/
python do_CV_split.py -n 3 -d your_datapath/ -s your_savepath/ -v
```
This script will create a cross-validation split with three folds (-n 3).

**5.** Train a protein language model (PLM)
Note: in this project, we decided to retrain a PLM. However, if you do not have access to GPUs, you can extract the BERT representation of proteins (2.), which uses [TAPE](https://github.com/songlab-cal/tape). If you have access to GPUs, you can retrain a PLM which will be further fine-tuned in the final model (7.).    
To do so, you can dowload [here](https://www.uniprot.org/downloads) the Swiss-prot data and then prepare the training-validation split to train the model:
```
cd experiments/
python do_process_swissprot.py -d your_datapath/ -s your_savepath/ -m 1000 -v
```
Where the -m is to limit the maximum size of proteins in number of amino acids. To create the training and validation split:
```
cd experiments/
python do_split_pretraining.py -f filename -s your_savepath/ -t 0.95 -v
```
Where t is the ratio of data used for training (here 95% of the data will be used for training, and 5% for validation). Provide the full path to your data, with the filename included (-f filename). 
And then, to train the PLM:
```
cd experiments/
python do_plm.py -c path/to/your/configfile.ini
```
This script will train a PLM according to the parameters defined in you configfile.ini. An example of a configfile can be found in *plm_configfiles/*

**6.** Train a chemical language model (CLM)
After downloading the training data from ChEMBL [here](https://www.ebi.ac.uk/chembl/), you can run the following script to prepare the training-validation split:
```
cd experiments/
python do_split_pretraining.py -f filename -s your_savepath/ -t 0.95 -v
```
Where t is the ratio of data used for training (here 95% of the data will be used for training, and 5% for validation). Provide the full path to your data, with the filename included (-f filename).     
And then, to train a CLM:
```
cd experiments/
python do_clm.py -c path/to/your/configfile.ini
```
You can find an example of a configfile in *clm_configfiles/*

**7.** Train a model to predict the interaction between proteins and molecules 
```
cd experiments/
python do_experiment -c path/to/your/configfile.ini -cv 0 -r 0
```
Will run an experiment with the parameters defined in the *.ini* file. The second argument is the CV fold (here, the model will run on the fold 0). You can, for example, run this in a for loop within a bash script to train the models at the same time if you have enoug GPUs. To train a model on all the data, use -cv 16. If you want to have statistical repeats, use the -r arguments with the number of repeats you want.   
The last model, and the losses are saved in *output/* under the configuration file name.   
You can find an example of a configfile in *experiment_configfiles/*

**8.** To use an ensemble of models to extract prediction, you can run locally:
```
cd experiments/
python do_ensemble_prediction.py -i 0 -n 10 -r path/to/you/data/ -m test
```
Where i is the chunk id you are running the prediction on, n the total number of chunks. In the root (-r), you should have the following folder: *data_clean/* which contains the dataset to run the prediction on and *A02_5repeats/* which contains the deep learning ensemble.     
You can find an example of a configfile in *experiment_deploy_configfiles/*, which is different from the example in *experiment_configfiles/* as you have to provide the path to the save pretrained model.

## Acknowledgements
We would like to thank the organizers of the challenge and the COVID-19 HPC Consortium, which allocated us the computational resources necessary to carry this project. More specifically, we would like to thank the Pittsburgh Supercomputer Center and Julian Uran for helping us to conduct our experiments.   
Finally, we would like to thank Francesca Grioni and Cyrill Brunner for their helpful discussions.
