import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import json


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, List, Tuple


def visualize_sparse_vectors(
    embeddings: np.ndarray,
    dim_min: int = 0,
    dim_max: int = 2731,
    show: bool = False,
    out_path: Optional[Path] = None,
    labeled_indices: Optional[List[Tuple[str, int]]] = None,
):
    """
    Visualizes embedding vectors as a horizontal color-coded heatmap.

    - Red for values close to 2
    - White for values close to 0
    - Blue for values close to -2

    Args:
        embeddings (np.ndarray): A NumPy array of shape (n, 2731) containing the embedding vectors.
        dim_min (int): The starting dimension index (inclusive).
        dim_max (int): The ending dimension index (exclusive).
        embedding_count (Optional[int]): Number of embeddings to visualize.
        show (bool): Whether to show the plot.
        out_path (Optional[Path]): Path to save the figure.
        labeled_indices (Optional[List[Tuple[str, int]]]): List of (label, index) pairs for marking specific dimensions.
    """
    if embeddings.ndim != 2:
        raise ValueError("Input must be a 2D array with shape (n, 768)")

    if not (0 <= dim_min < dim_max <= embeddings.shape[1]):
        raise ValueError(f"dim_min and dim_max must be within [0, {embeddings.shape[1]}] and dim_min < dim_max")

    embeddings = embeddings[:, dim_min:dim_max]

    # norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    # fig, ax = plt.subplots(figsize=(12, embeddings.shape[0] * 0.5))
    # cax = ax.imshow(embeddings, cmap="hot", norm=norm, aspect="auto")

    norm = mcolors.Normalize(vmin=0, vmax=2)  # Linear gradient from 0 to 2
    fig, ax = plt.subplots(figsize=(12, embeddings.shape[0] * 0.5))

    custom_cmap = mcolors.LinearSegmentedColormap.from_list("white_to_blue", ["white", "red"])
    cax = ax.imshow(embeddings, cmap=custom_cmap, norm=norm, aspect="auto")

    ax.set_xlabel("Embedding Dimension Index")
    ax.set_ylabel("Document Index")

    tick_positions = np.linspace(0, dim_max - dim_min, num=5)
    tick_labels = np.round(np.linspace(dim_min, dim_max, num=5)).astype(int)

    if labeled_indices:
        for label, index in labeled_indices:
            if dim_min <= index < dim_max:
                tick_positions = np.append(tick_positions, index - dim_min)
                tick_labels = np.append(tick_labels, label)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_yticks(range(embeddings.shape[0]))
    ax.set_yticklabels(range(embeddings.shape[0]))

    cbar = fig.colorbar(cax, orientation="vertical", shrink=0.5)
    cbar.set_label("Embedding Value")

    ax.set_title("Sparse Vector Heatmap")

    if out_path:
        print(f"Saving figure to {out_path}")
        # plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()

    plt.close(fig)


def get_regions(min_val, max_val, divisions):
    # Calculate the step size for the specified number of divisions
    step = (max_val - min_val) // divisions

    # Generate regions as tuples of start and end indices
    for i in range(divisions):
        start = min_val + i * step
        end = start + step
        yield (start, end)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Illustrative Examples of Sparse Vector")
    parser.add_argument("glosses", type=str, nargs="+", help="List of gloss values for analysis")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--embedding_count", type=int, default=1)
    parser.add_argument("--dim_min", type=int, default=0)
    parser.add_argument("--dim_max", type=int, default=2731)
    parser.add_argument(
        "--out_folder", type=Path, default=Path(r"C:\Users\Colin\projects\PhD\PhD Proposal\sparse_vector_illustration")
    )
    args = parser.parse_args()

    glosses = args.glosses

    # https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/word2vec.py

    # create fake sparse vector
    sparse_vector = np.zeros((args.embedding_count, args.dim_max))

    # Randomly assign each string label to a unique column index
    # np.random.seed(42)  # For reproducibility
    available_columns = np.random.permutation(args.dim_max)[: len(glosses)]
    glosses = sorted(glosses)
    available_columns = sorted(available_columns)
    gloss_indices = list(zip(glosses, available_columns))

    # gloss_indices = [("BLACK", 1), ("RUSSIA", 700)]

    # Assign random values between -2 and 2 (inclusive) at specified indices
    for row_index in range(sparse_vector.shape[0] - 1):
        for _, col_index in gloss_indices:
            sparse_vector[row_index, col_index] = np.random.uniform(0, 0.8)

    # set the last one to be all 2s
    for _, col_index in gloss_indices:
        sparse_vector[-1, col_index] = 2

    if args.out_folder:
        out_dir = args.out_folder / "_".join(glosses)
        out_dir.mkdir(exist_ok=True)
        glosses_str = "_".join(glosses)
        out_file_path = out_dir / f"{glosses_str}.png"
    else:
        out_file_path = None

    visualize_sparse_vectors(
        sparse_vector,
        dim_min=args.dim_min,
        dim_max=args.dim_max,
        show=True,
        labeled_indices=gloss_indices,
        out_path=out_file_path,
    )
