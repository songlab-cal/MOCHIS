# Tabula Muris Senis Analysis

Here, we organize our analyses and results obtained from applying MOCHIS to [_Tabula Muris Senis_](https://tabula-muris-senis.ds.czbiohub.org/). 

## Raw Data 

The scRNA-seq data are publicly available at [this link](https://cellxgene.cziscience.com/collections/0b9d8a04-bb9d-44da-aa27-705bb65b54eb).

Concretely, we downloaded Smartseq-2 assays (stored as `local.h5ad`) for each tissue. 

## Analysis Results

Please find the following files in this directory.

- Mann-Whitney differentially expressed genes (at FDR control of 0.05): `tissues/mw_sig_3m_18m.csv`, `tissues/mw_sig_18m_24m.csv`, `tissues/mw_sig_24m_3m.csv`  
- MOCHIS differentially expressed genes (at FDR control of 0.05): `tissues/mochis_sig_3m_18m.csv`, `tissues/mochis_sig_18m_24m.csv`, `tissues/mochis_sig_24m_3m.csv`
- Tissue-specific tables of _p_-values: `/tissues/[NAME OF TISSUE]`   

## Detailed Analysis

Please see `mochis_vs_mw.ipynb` and its accompanying RMarkdown file `mochis_vs_mw.Rmd`.
