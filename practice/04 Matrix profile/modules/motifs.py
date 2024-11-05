import numpy as np
import stumpy

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    # INSERT YOUR CODE
    
    matrix_profile_array = matrix_profile['mp'].copy()
    exclusion_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        # Find the minimum distance in the matrix profile array
        min_idx = np.argmin(matrix_profile_array)
        min_distance = matrix_profile_array[min_idx]

        # Append motif information
        motifs_dist.append(min_distance)
        motifs_idx.append((min_idx, min_idx + matrix_profile['m']))  # Correct calculation of the right index

        # Apply exclusion zone around the found motif
        apply_exclusion_zone(matrix_profile_array, min_idx, exclusion_zone, np.inf)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
