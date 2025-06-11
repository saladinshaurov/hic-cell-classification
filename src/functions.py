import hdbscan, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

def prepare_data (location: str) -> tuple:
    """Return pandas dataframe with proper labeled columns and adjusted
    values.

    Args:
        location (str): The location directory in string form.

    Returns:
        pd.DataFrame: data file -> pandas dataframe
    """
    cols = ['chr_i', 'pos_i', 'chr_j', 'pos_j', 'contacts']
    
    df = pd.read_csv(location, names=cols, sep=r'\s+')
    df['pos_i'] = df['pos_i'] // 1000000
    df['pos_j'] = df['pos_j'] // 1000000
    return df

def find_chr_length (data: pd.DataFrame) -> dict[str: tuple]:
    """Return a dictionary containing each unique chromosome mapped to their lengths
    as a tuple (min_length, max_length).
    
    Args:
        data (pd.DataFrame): DataFrame with ['chr_i', 'pos_i', 'chr_j', 'pos_j', 'contacts']

    Returns:
        dict: chromosome -> length (min, max)
    """
    chr_bounds = {}
    chromosomes = sorted(set(df['chr_i']).union(df['chr_j']), key=lambda x: int(x[3:]) if x[3:].isdigit() else 100)
    for chr_ in chromosomes:
            max_i = data.loc[data['chr_i'] == chr_, 'pos_i'].max()
            max_j = data.loc[data['chr_j'] == chr_, 'pos_j'].max()
            min_i = data.loc[data['chr_i'] == chr_, 'pos_i'].min()
            min_j = data.loc[data['chr_j'] == chr_, 'pos_j'].min()
            chr_bounds[chr_] = (min(min_i, min_j), max(max_i, max_j))
    return chr_bounds

def build_chr_block_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame where rows and columns are chromosomes and
    each cell contains a symmetric contact matrix (DataFrame) between those chromosomes.

    Args:
        data (pd.DataFrame): DataFrame with ['chr_i', 'pos_i', 'chr_j', 'pos_j', 'contacts']

    Returns:
        pd.DataFrame: block DataFrame of contact matrices
    """
    # compute length of chromosome
    chr_bounds = find_chr_length(data)
    chromosomes = sorted(set(data['chr_i']).union(data['chr_j']), key=lambda x: int(x[3:]) if x[3:].isdigit() else 100)

    # initialize zero block matrix as pandas dataframe
    chr_block_df = pd.DataFrame(0, index=chromosomes, columns=chromosomes, dtype=object)

    # fill diagonal with contact matrices between chromosomes
    for chr_i in chromosomes:
        for chr_j in chromosomes:
            # set lengths for chromosome contact matrix
            min_i, max_i = chr_bounds[chr_i]
            min_j, max_j = chr_bounds[chr_j]
            row_bins = list(range(min_i, max_i + 1))
            col_bins = list(range(min_j, max_j + 1))

            # find the diagonal point
            if chr_i == chr_j:
                # initialize zero matrix of size chromosome
                mat = pd.DataFrame(0, index=row_bins, columns=col_bins)
                # find subset of actual contact data for this chr pair
                subdf = data[(data['chr_i'] == chr_i) & (data['chr_j'] == chr_j)]
                for _, row in subdf.iterrows():
                    i, j, c = row['pos_i'], row['pos_j'], row['contacts']
                    mat.iat[i - min_i, j - min_j] = c
                # set diagonal block as that contact matrix
                chr_block_df.at[chr_i, chr_j] = mat
    return chr_block_df

def cell_summary_stats(matrix: pd.DataFrame) -> dict:
    """
    Return summary statistics of the cell based on intra-chromosomal contact matrices
    (i.e., the diagonal blocks only).

    Args:
        matrix (pd.DataFrame): block matrix where only diagonal cells contain data

    Returns:
        dict: {
            'total_contacts': float,
            'mean_contact': float,
            'var_contact': float
        }
    """
    all_contacts = []

    for chr in matrix.index:
        block = matrix.at[chr, chr]
        # convert block as numpy array and then flaten into 1D array and append
        all_contacts.append(block.values.ravel())

    # transform all 1D arrays into one 1D array
    flat_contacts = np.concatenate(all_contacts)

    num_zero = np.sum(flat_contacts == 0)
    num_nonzero = np.sum(flat_contacts > 0)

    return {
        'total_contacts': float(np.sum(flat_contacts)),
        'mean_contact': float(np.mean(flat_contacts)),
        'var_contact': float(np.var(flat_contacts)),
        'num_zero_contacts': int(num_zero),
        'num_nonzero_contacts': int(num_nonzero)
    }

def per_chr_summary_stats(df: pd.DataFrame, matrix: pd.DataFrame) -> dict:
    """
    For each chromosome's intra-chromosomal contact matrix (on the diagonal),
    return summary stats including:
      - Total, mean, variance of contact intensity
      - Total, mean, variance of diagonal-only entries
      - Mean and variance of distance-from-diagonal (|i - j|) for non-zero contacts

    Args:
        df (pd.DataFrame): not used
        matrix (pd.DataFrame): 23x23 block matrix with only diagonal blocks populated

    Returns:
        dict: {
            'chrX': {
                'total': float,
                'mean': float,
                'var': float,
                'diag_total': float,
                'diag_mean': float,
                'diag_var': float,
                'contact_dist_mean': float,
                'contact_dist_var': float
            },
            ...
        }
    """
    stats = {}

    for chr in matrix.index:
        block = matrix.at[chr, chr]

        # convert blocks and giagonal into numpy arrays
        values = block.values
        diag_values = np.diag(values)

        # Get distance-from-diagonal (abs(i - j)) for non-zero contacts
        contact_dists = []
        # get the row dimension of array
        n = values.shape[0]

        # iterate through numpy array
        for i in range(n):
            for j in range(n):
                contact = values[i, j]
                if contact > 0:
                    dist = abs(i - j)
                    contact_dists.append(dist)

        # convert list into numpy array
        contact_dists = np.array(contact_dists)

        stats[chr] = {
            'total': float(np.sum(values)),
            'mean': float(np.mean(values)),
            'var': float(np.var(values)),

            'diag_total': float(np.sum(diag_values)),
            'diag_mean': float(np.mean(diag_values)),
            'diag_var': float(np.var(diag_values)),

            'contact_dist_mean': float(np.mean(contact_dists)) if len(contact_dists) > 0 else 0.0,
            'contact_dist_var': float(np.var(contact_dists)) if len(contact_dists) > 0 else 0.0
        }

    return stats

def read_all_in_cell_type(folder_path: str, cell_type: str, file_suffix: str = ".txt") -> pd.DataFrame:
    """
    Loop through all cell files in a folder and extract cell-level and
    per-chromosome contact statistics.

    Args:
        folder_path (str): Path to directory with cell contact files.
        file_suffix (str): File extension to filter on (default = ".txt").

    Returns:
        pd.DataFrame: One row per cell, with summary statistics as columns.
    """
    cells = []

    for fname in os.listdir(folder_path):
        cell_id = os.path.splitext(fname)[0]
        cell_path = os.path.join(folder_path, fname)

        try:
            # Load and process each cell data
            df = prepare_data(location=cell_path)
                
            matrix = build_chr_block_matrix(df)

            features = {'cell_id': cell_id, 'cell_type': cell_type}

            # add summary stats for whole cell as features
            features.update(cell_summary_stats(matrix))

            # add summary stats by chromosome for each cell
            per_chr_stats = per_chr_summary_stats(df, matrix)
            for chr_name, chr_stat_dict in per_chr_stats.items():
                for stat_name, value in chr_stat_dict.items():
                    features[f"{chr_name}_{stat_name}"] = value

            # add features for this cells as a dictionary
            cells.append(features)

        except Exception as error:
            print(f"Error processing {fname}: {error}")

    return pd.DataFrame(cells).set_index('cell_id')

def keep_top_75_percent(group):
    threshold = group["num_zero_contacts"].quantile(0.75)
    return group[group["num_zero_contacts"] <= threshold]

def extract_upper_triangle_features(matrix: pd.DataFrame) -> np.ndarray:
    """
    Extracts and concatenates the upper triangle (excluding diagonal)
    from all diagonal contact matrices in a block matrix.
    
    Returns a 1D feature vector.
    """
    features = []
    for chr in matrix.index:
        block = matrix.at[chr, chr]
        if isinstance(block, pd.DataFrame):
            values = block.values
            triu_indices = np.triu_indices_from(values, k=1)  # exclude diagonal
            upper_triangle = values[triu_indices]
            features.append(upper_triangle)
    return np.concatenate(features)