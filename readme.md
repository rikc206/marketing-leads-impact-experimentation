# DMA Pairs Analysis Script

## Overview
This Python script (20250303_DMA_Pairs_Python_Script.py) analyzes DMA (Designated Market Area) leads and promotional data for marketing campaign analysis. The script performs:

- Data merging between DMA leads and promotional calendar data
- Analysis of key features like:
  - Digital spend (Meta + Google)
  - Digital spend per lead 
  - Non-digital leads
- Statistical analysis including:
  - Variance Inflation Factor (VIF) to check feature multicollinearity
  - Kolmogorov-Smirnov tests to compare DMA distributions
  - Seasonal-trend decomposition (STL) to analyze time series patterns
- Results export to Excel files in the Outputs folder

## Required Files
- Input files (in Inputs folder):
  - EQX DMA Leads 1.24.25.xlsx: DMA leads data
  - Promo Calendar Features.xlsx
  - Promotional Calendar data.xlsx
- Output files (generated in Outputs folder):
  - VIF_Results.xlsx
  - {column}_STL_Correlations.xlsx 
  - {column}_KS_Test_Results.xlsx

## Dependencies
Required Python packages:
- numpy
- pandas 
- scipy
- statsmodels

Install dependencies using:
pip install -r requirements.txt