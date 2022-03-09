# Single-Cell-Coupled-Autoencoders

In this project, we implement a coupled autoencoder for working with single-cell data, which includes data sets on DNA, mRNA and protein data.

## About

Historically, analysis on single-cell data has been difficult to perform, due to data collection methods often resulting in the destruction of the cell in the process of collecting information. However, an ongoing endeavor of biological data science has recently been to analyze different modalities, or forms, of the genetic information within a cell. Doing so will allow modern medicine a greater understanding of cellular functions and how cells work in the context of illnesses. The information collected on the three modalities of DNA, RNA, and protein can be done safely and because it is known that they are same information in different forms, analysis done on them can be extrapolated understand the cell as a whole. Previous research has been conducted by Gala, R., Budzillo, A., Baftizadeh, F. et al. to capture gene expression in neuron cells with a neural network called a coupled autoencoder. This autoencoder framework is able to reconstruct the inputs, allowing the prediction of one input to another, as well as align the multiple inputs in the same low dimensional representation. In our paper, we build upon this coupled autoencoder on a data set of cells taken from several sites of the human body, predicting from RNA information to protein. We find that the autoencoder is able to adequately cluster the cell types in its lower dimensional representation, as well as perform decently at the prediction task. We show that the autoencoder is a powerful tool for analyzing single-cell data analysis and may prove to be a valuable asset in single-cell data analysis.

## Instructions for running this project

1. Clone this repository onto your local machine with `git clone https://github.com/18anguyen9/Single_Cell_Coupled_Autoencoders.git` and change into the directory.

2. Launch the Docker image for the project with the following line: `launch.sh -i alandnin/method3:latest`

3. The code is ran with command line arguments:

    * `test`: (`python3 run.py test`) Due to the size of the full data set, this will perform a test run on a much smaller subset of our data sets to simulate the output of our project. 
    
    *  `test-full`: (`python3 run.py test-full`) This a full run of training the coupled autoencoder with the entire data set. This will take much longer (expect 20-30 minutes) than `test`, but will contain meaningful outputs compared to the simulated `test`. However, you will first need to download the entire data set using the following command line argument `aws s3 sync s3://openproblems-bio/public/ $HOME/data/ --no-sign-request` and moving the `cite_gex_processed_training.h5ad` and `cite_adt_processed_training.h5ad` files into the `/data` file of this directory.
    
    *  `clear-cache`: The training computation is memory expensive. `python3 run.py clear-cache` can be run in the case that your machine runs out of memory. This will only happen when `test-full` is ran.

## Related

We based our project off of this NeurIPS competition: https://openproblems.bio/neurips_docs/about/about/

Our website: https://18anguyen9.github.io/DSC_180_website/
    
    
    
