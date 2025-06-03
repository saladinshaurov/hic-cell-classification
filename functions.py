import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

def create_chr_contact_matrix_df(data: pd.DataFrame, bin_size: int = 1) -> pd.DataFrame:
    """
    Create a DataFrame where rows and columns are chromosomes and
    each cell contains a symmetric contact matrix (DataFrame) between those chromosomes.

    Args:
        data (pd.DataFrame): DataFrame with ['chr_i', 'pos_i', 'chr_j', 'pos_j', 'contacts']
        bin_size (int): resolution for contact bins (default = 1 for base-level; set higher for binning)

    Returns:
        pd.DataFrame: block DataFrame of contact matrices
    """
    # Step 1: Compute min/max per chromosome
    chr_bounds = {}
    chromosomes = sorted(set(data['chr_i']).union(data['chr_j']))

    for chr_ in chromosomes:
        max_i = data.loc[data['chr_i'] == chr_, 'pos_i'].max()
        max_j = data.loc[data['chr_j'] == chr_, 'pos_j'].max()
        min_i = data.loc[data['chr_i'] == chr_, 'pos_i'].min()
        min_j = data.loc[data['chr_j'] == chr_, 'pos_j'].min()
        chr_bounds[chr_] = (min(min_i, min_j), max(max_i, max_j))

    # Step 2: Initialize the block matrix
    chr_block_df = pd.DataFrame(index=chromosomes, columns=chromosomes, dtype=object)

    # Step 3: Fill each block
    for chr_i in chromosomes:
        for chr_j in chromosomes:
            # Get bin coordinates
            min_i, max_i = chr_bounds[chr_i]
            min_j, max_j = chr_bounds[chr_j]
            row_bins = list(range(min_i, max_i + 1, bin_size))
            col_bins = list(range(min_j, max_j + 1, bin_size))

            # Initialize zero matrix
            mat = pd.DataFrame(0, index=row_bins, columns=col_bins)

            # Subset actual contact data for this chr pair
            subdf = data[(data['chr_i'] == chr_i) & (data['chr_j'] == chr_j)]

            for _, row in subdf.iterrows():
                i, j = row['pos_i'], row['pos_j']
                mat.at[i, j] = row['contacts']
                # Add symmetric entry if intra-chromosomal
                if chr_i == chr_j:
                    mat.at[j, i] = row['contacts']

            chr_block_df.at[chr_i, chr_j] = mat

    return chr_block_df


def chrSize (data: pd.DataFrame) -> dict[str: tuple[int]]:
    """Return a dictionary containing keys: each unique chromosome and
    values: tuple(minimum size of chromosome, maxiumum size of chromosome) 

    Args:
        data (pd.DataFrame): _description_
    """
    chr_set = set(data['chr_i']).union(data['chr_i'])
    
    result = {}
    for chr_ in chr_set:
        max_size_i = int(data.loc[data['chr_i'] == chr_, 'pos_i'].max())
        max_size_j = int(data.loc[data['chr_j'] == chr_, 'pos_j'].max())
        
        min_size_i = int(data.loc[data['chr_i'] == chr_, 'pos_i'].min())
        min_size_j = int(data.loc[data['chr_j'] == chr_, 'pos_j'].min())
        
        result[chr_] = (min(min_size_i, min_size_j), max(max_size_i, max_size_j)) 

    return result
    
def prepareData (location: str) -> pd.DataFrame:
    """Return pandas dataframe with proper labeled columns and adjusted
    values.

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    cols = ['chr_i', 'pos_i', 'chr_j', 'pos_j', 'contacts']
    df = pd.read_csv(location, names=cols, sep='\s+')
    df['pos_i'] = df['pos_i'] // 1000000
    df['pos_j'] = df['pos_j'] // 1000000
    
    return df
    
df = prepareData('/Users/salah/Documents/chromosome-prediction/data/ml3_AAGCGACC-ACCTCTTG.txt')
matrix = create_chr_contact_matrix_df(df)

sns.heatmap(matrix.at['chr3', 'chr3'], cmap='YlGnBu')
#sns.heatmap(coor_matrix, cmap='YlGnBu')
plt.tight_layout()
plt.show()