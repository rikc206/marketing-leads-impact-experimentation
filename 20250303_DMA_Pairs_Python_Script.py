"""
This module analyzes relationships between pairs of DMAs (Designated Market Areas).

It performs the following analyses:
1. Merges promotional calendar data with DMA leads data
2. Calculates Variance Inflation Factor (VIF) to check feature multicollinearity
3. Compares DMA distributions using Kolmogorov-Smirnov tests
4. Analyzes time series patterns using Seasonal-Trend decomposition (STL)
5. Calculates correlations between DMA pairs for trends and seasonality

Results are exported to Excel files for further analysis.
"""

# Standard library imports
import itertools
import warnings

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor,
)
import statsmodels.api as sm

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Constant Variables
EQX_DMA_LEADS_PATH = "Inputs/DMA Leads 1.24.25.xlsx"
PROMO_FEATURES_PATH = "Inputs/Promo Calendar Features.xlsx"
DATE_RANGE_START = "2023-01-01"
DATE_RANGE_END = "2024-12-31"
VIF_OUTPUT_PATH = "Outputs/VIF_Results.xlsx"


# Read DMA leads data from Excel
leads_df = pd.read_excel(
    EQX_DMA_LEADS_PATH,
    skiprows=1,
    usecols="B:G"
)

# Read promotional calendar features data
promo_features_df = pd.read_excel(
    PROMO_FEATURES_PATH, 
    skiprows=1,
    usecols="B:J"
)

# Filter out dmas that are not required
excluded_dmas = ['Toronto', 'Vancouver', 'London']
leads_df = leads_df[~leads_df['DMA'].isin(excluded_dmas)]

# Calculate derived features for spend and leads analysis
leads_df['Digital_Spend'] = leads_df['Meta Spend'] + leads_df['Google Spend']
leads_df['Non_Digital_Leads'] = leads_df['Total Leads'] - leads_df['Digital Leads']
leads_df['Digital_Spend_per_Total_Lead'] = (
    leads_df['Digital_Spend'] / leads_df['Total Leads']
)
leads_df['Digital_Spend_per_Digital_Lead'] = (
    leads_df['Digital_Spend'] / leads_df['Digital Leads']
)


def fill_missing_dates(df: pd.DataFrame, dma: str) -> pd.DataFrame:
    """
    Fill missing dates in timeseries data with zeros.

    Args:
        df: Input DataFrame with DMA data.
        dma: DMA region name.

    Returns:
        DataFrame with complete date range and filled values.
    """
    df_filled = df.copy()
    date_range = pd.date_range(
        start=DATE_RANGE_START, end=DATE_RANGE_END, freq='D'
    )
    existing_dates = pd.to_datetime(df_filled['Create Date'].unique())
    missing_dates = set(date_range) - set(existing_dates)

    if missing_dates:
        missing_df = pd.DataFrame({'Create Date': list(missing_dates)})
        missing_df['DMA'] = dma

        # Fill numeric columns with 0
        numeric_cols = [
            col for col in df_filled.columns if col not in ['Create Date', 'DMA']
        ]
        for col in numeric_cols:
            missing_df[col] = 0

        df_filled = (
            pd.concat([df_filled, missing_df], ignore_index=True)
            .sort_values('Create Date')
            .reset_index(drop=True)
        )

    return df_filled


def merge_promotions_data(
    leads_data: pd.DataFrame, promo_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge initial data with promotions data by matching dates.

    Args:
        leads_data: Contains lead count at a DMA and date level.
        promo_data: DataFrame containing promotions data with dates.

    Returns:
        DataFrame with merged promotion features.
    """
    df_merged = leads_data.copy()

    # Initialize promotion columns with None
    promo_cols = ['Init', 'Init Back', 'GC/Cashback', 'On us', 'Details']
    for col in promo_cols:
        df_merged[col] = None

    # Match promotions with dates and merge data
    for idx, row in df_merged.iterrows():
        matching_promos = promo_data[
            (promo_data['Start Date'] <= row['Create Date']) &
            (promo_data['End Date'] >= row['Create Date'])
        ]
        if not matching_promos.empty:
            promo = matching_promos.iloc[0]
            for col in promo_cols:
                df_merged.at[idx, col] = promo[col]

    # Convert promotion columns to numeric where applicable
    numeric_promo_cols = ['Init', 'Init Back', 'GC/Cashback', 'On us']
    for col in numeric_promo_cols:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

    return df_merged


def calculate_vif(df: pd.DataFrame, dma: str) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for selected numeric features in a DMA.

    Args:
        df: Input DataFrame contaning merged data of leads and promo features.
        dma: DMA region to analyze.

    Returns:
        DataFrame with VIF values for each numeric feature in the specified DMA.
    """
    # Features columns on which we need to do VIF
    feature_cols = [
        'Create Date', 'DMA', 'Init', 'Init Back', 'GC/Cashback',
        'On us', 'Total Leads', 'Digital_Spend_per_Digital_Lead'
    ]
    dma_data = df[df['DMA'] == dma][feature_cols].copy()

    # Get numeric columns for VIF calculation
    numeric_features = dma_data.select_dtypes(include=['float64', 'int64']).columns
    clean_data = dma_data[numeric_features]

    # Handle invalid values through interpolation
    clean_data = (
        clean_data.replace([np.inf, -np.inf], np.nan)
        .interpolate(method='linear')
        .fillna(method='bfill')
        .fillna(method='ffill')
    )

    # Calculate VIF scores
    features_with_constant = sm.add_constant(clean_data)
    vif_values = [
        variance_inflation_factor(features_with_constant.values, i)
        for i in range(1, features_with_constant.shape[1])
    ]

    return pd.DataFrame({
        "DMA": dma,
        "Variable": numeric_features,
        "VIF": vif_values,
    })


def compare_dma_distributions(dma1: str, dma2: str, column: str) -> pd.DataFrame:
    """
    Perform Kolmogorov-Smirnov test between two DMAs for a given column.

    Args:
        dma1: First DMA name.
        dma2: Second DMA name.
        column: Column name to compare.

    Returns:
        DataFrame with test statistics and p-value.
    """
    dma1_values = leads_df[leads_df['DMA'] == dma1][column]
    dma2_values = leads_df[leads_df['DMA'] == dma2][column]
    ks_stat, p_val = stats.ks_2samp(dma1_values, dma2_values)

    return pd.DataFrame({
        'DMA1': [dma1],
        'DMA2': [dma2],
        'ks_statistic': [ks_stat],
        'P_value': [p_val]
    })


def compare_stl_components(
    dma1: str, dma2: str, column: str, period: int
) -> pd.DataFrame:
    """
    Compare seasonal-trend decomposition between two DMAs.

    Args:
        dma1: First DMA name.
        dma2: Second DMA name.
        column: Column name to analyze.
        period: Number of periods for seasonal decomposition.

    Returns:
        DataFrame with correlation metrics between components.
    """
    def prepare_timeseries(dma: str) -> pd.Series:
        """
        Prepare time series data for STL decomposition.

        Args:
            dma: DMA name to prepare data for.

        Returns:
            Processed time series with handled missing/invalid values.
        """
        series = (
            leads_df[leads_df['DMA'] == dma][['Create Date', column]]
            .copy()
            .pipe(fill_missing_dates, dma)
            .set_index('Create Date')[column]
        )
        series = (
            series.replace([np.inf, -np.inf], np.nan)
            .interpolate(method='linear')
            .fillna(method='bfill')
            .fillna(method='ffill')
        )
        return series

    series1 = prepare_timeseries(dma1)
    series2 = prepare_timeseries(dma2)

    try:
        stl1 = STL(series1, period=period).fit()
        stl2 = STL(series2, period=period).fit()

        trend_correlation = np.corrcoef(stl1.trend, stl2.trend)[0, 1]
        seasonal_correlation = stats.spearmanr(stl1.seasonal, stl2.seasonal)[0]
        total_correlation = stats.spearmanr(series1, series2)[0]

        return pd.DataFrame({
            'DMA1': [dma1],
            'DMA2': [dma2],
            'trend_correlation': [trend_correlation],
            'seasonal_correlation': [seasonal_correlation],
            'total_correlation': [total_correlation]
        })
    except Exception as e:
        print(f"STL decomposition failed for {dma1} and {dma2}: {e}")
        return None


if __name__ == '__main__':
    # Merge promotional data with leads data
    lead_promo_merged_df = merge_promotions_data(leads_df, promo_features_df)

    # Calculate and save VIF results for each DMA
    dma_list = lead_promo_merged_df['DMA'].unique()
    vif_results = pd.concat([
        calculate_vif(lead_promo_merged_df, dma) for dma in dma_list
    ])
    vif_results.to_excel(VIF_OUTPUT_PATH, index=False)

    # Generate all possible DMA pairs for comparison
    dma_pairs = list(itertools.combinations(leads_df['DMA'].unique(), 2))
    analysis_columns = [
        'Total Leads',
        'Digital_Spend_per_Digital_Lead',
        'Non_Digital_Leads'
    ]

    # Perform analysis for each metric
    for column in analysis_columns:
        # Calculate and save STL correlations
        stl_results = pd.concat([
            compare_stl_components(dma1, dma2, column, period=7)
            for dma1, dma2 in dma_pairs
        ])
        stl_output_path = f"Outputs/{column}_STL_Correlations.xlsx"
        stl_results.to_excel(stl_output_path, index=False)

        # Calculate and save KS test results
        ks_results = pd.concat([
            compare_dma_distributions(dma1, dma2, column)
            for dma1, dma2 in dma_pairs
        ])
        ks_output_path = f"Outputs/{column}_KS_Test_Results.xlsx"
        ks_results.to_excel(ks_output_path, index=False)
