# Tabula Muris Senis Analysis

Here, we organize our analyses and results obtained from applying MOCHIS to [_Tabula Muris Senis_](https://tabula-muris-senis.ds.czbiohub.org/). 

## Raw Data 

The scRNA-seq data are publicly available at [this link](https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb).

Concretely, we downloaded Smartseq-2 assays (stored as `local.rds`) for each tissue. 

## Analysis Results

Please find the following files in this directory.

- Mann-Whitney differentially expressed genes (at FDR control of 0.05): `results/MW` 
- MOCHIS differentially expressed genes (at FDR control of 0.05): `results/MOCHIS`
- Tissue-specific tables of _p_-values: `results/tissues/[NAME OF TISSUE]`   
- Figures from analysis: `results/figures`

## Detailed Analysis

Please see `mochis_vs_mw.pdf` and its accompanying RMarkdown file `mochis_vs_mw.Rmd`.

