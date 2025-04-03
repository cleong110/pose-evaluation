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


def visualize_multiple_embeddings(
    gloss_embeddings: List[Tuple[str, np.ndarray]],
    dim_min: int = 0,
    dim_max: int = 768,
    embedding_count: Optional[int] = None,
    show: bool = False,
    out_path: Optional[Path] = None,
):
    """
    Visualizes multiple sets of embedding vectors as a stacked color-coded heatmap.

    - Red for values close to 2
    - White for values close to 0
    - Blue for values close to -2

    Args:
        gloss_embeddings (List[Tuple[str, np.ndarray]]): A list of (gloss, embeddings) pairs.
        dim_min (int): The starting dimension index (inclusive).
        dim_max (int): The ending dimension index (exclusive).
        embedding_count (Optional[int]): Maximum number of embeddings per gloss.
        show (bool): Whether to display the plot.
        out_path (Optional[Path]): Path to save the figure.
    """

    all_embeddings = []
    gloss_labels = []
    y_tick_positions = []
    current_y = 0

    for gloss, embeddings in gloss_embeddings:
        if embeddings.ndim != 2 or embeddings.shape[1] != 768:
            raise ValueError(f"Embeddings for gloss '{gloss}' must be a 2D array with shape (n, 768)")

        if embedding_count is not None:
            embeddings = embeddings[:embedding_count]

        embeddings = embeddings[:, dim_min:dim_max]

        all_embeddings.append(embeddings)
        gloss_labels.append(gloss)

        # Record the y-tick position for labeling
        mid_y = current_y + embeddings.shape[0] / 2
        y_tick_positions.append(mid_y)

        # Update the current y position for stacking
        current_y += embeddings.shape[0]

    # Stack all embeddings vertically
    stacked_embeddings = np.vstack(all_embeddings)

    # Normalize to range [-2, 2] for colormap scaling
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, len(stacked_embeddings) * 0.2))

    # Use imshow for visualization
    cax = ax.imshow(stacked_embeddings, cmap="bwr", norm=norm, aspect="auto")

    # Set axis labels
    ax.set_xlabel("Embedding Dimension Index")
    ax.set_ylabel("Gloss")

    # Set x-axis ticks to show actual embedding dimensions
    ax.set_xticks(np.linspace(0, dim_max - dim_min, num=5))  # Show around 5 ticks
    ax.set_xticklabels(np.round(np.linspace(dim_min, dim_max, num=5)).astype(int))

    # Set y-axis ticks with gloss labels
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(gloss_labels)

    # Add color bar
    cbar = fig.colorbar(cax, orientation="vertical", shrink=0.5)
    cbar.set_label("Embedding Value")

    if show:
        plt.show()
    if out_path:
        print(f"Saving figure to {out_path}")
        plt.savefig(out_path, bbox_inches="tight")


def visualize_embeddings_for_multiple_glosses(
    df,
    model,
    gloss_list: List[str],
    embedding_count: Optional[int] = None,
    dim_min: int = 0,
    dim_max: int = 768,
    show: bool = False,
    out_folder: Optional[Path] = None,
):
    """
    Loads and visualizes embeddings for multiple glosses.

    Args:
        df: DataFrame containing gloss and embedding file paths.
        model: Model name to filter the embeddings.
        gloss_list (List[str]): List of glosses to visualize.
        embedding_count (Optional[int]): Max number of embeddings per gloss.
        dim_min (int): Minimum embedding dimension to visualize.
        dim_max (int): Maximum embedding dimension to visualize.
        show (bool): Whether to show the plot.
        out_folder (Optional[Path]): Folder to save the figure.
    """
    df = df[df["Embedding Model"] == model]

    gloss_embeddings = []

    for gloss in gloss_list:
        gloss_df = df[df["Gloss"] == gloss]
        embeddings_list = [load_embedding(embed_path) for embed_path in gloss_df["Embedding File Path"].tolist()]
        embeddings = np.array(embeddings_list)

        gloss_embeddings.append((gloss, embeddings))

    if out_folder is not None:
        glosses_str = "_".join(glosses)
        if embedding_count is None:
            embedding_count = "all"
        out_path = (
            out_folder
            / f"{glosses_str}_{model}_embeddings_{embedding_count}_samples_mindim_{dim_min}_maxdim_{dim_max}_colorcoded.png"
        )
    else:
        out_path = None

    visualize_multiple_embeddings(
        gloss_embeddings,
        dim_min=dim_min,
        dim_max=dim_max,
        embedding_count=embedding_count,
        show=show,
        out_path=out_path,
    )


def visualize_embeddings(
    embeddings: np.ndarray,
    dim_min: int = 0,
    dim_max: int = 768,
    embedding_count: Optional[int] = None,
    show: bool = False,
    out_path: Optional[Path] = None,
    gloss: str = "",
):
    """
    Visualizes embedding vectors as a horizontal color-coded heatmap.

    - Red for values close to 2
    - White for values close to 0
    - Blue for values close to -2

    Args:
        embeddings (np.ndarray): A NumPy array of shape (n, 768) containing the embedding vectors.
        dim_min (int): The starting dimension index (inclusive).
        dim_max (int): The ending dimension index (exclusive).
    """
    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise ValueError("Input must be a 2D array with shape (n, 768)")

    if not (0 <= dim_min < dim_max <= 768):
        raise ValueError("dim_min and dim_max must be within [0, 768] and dim_min < dim_max")

    if embedding_count is not None:
        embeddings = embeddings[:embedding_count]

    # Slice the embeddings to the specified dimension range
    embeddings = embeddings[:, dim_min:dim_max]

    # Normalize to range [-2, 2] for colormap scaling
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, embeddings.shape[0] * 0.5))

    # Use imshow for visualization
    cax = ax.imshow(embeddings, cmap="bwr", norm=norm, aspect="auto")

    # Set axis labels
    ax.set_xlabel("Embedding Dimension Index")
    ax.set_ylabel(f"{gloss} Embedding Vector Index")

    # Set x-axis ticks to show actual embedding dimensions
    ax.set_xticks(np.linspace(0, dim_max - dim_min, num=5))  # Show around 5 ticks
    ax.set_xticklabels(np.round(np.linspace(dim_min, dim_max, num=5)).astype(int))

    # Set y-axis ticks to label the embedding vectors
    ax.set_yticks(range(embeddings.shape[0]))
    ax.set_yticklabels([f"{gloss} Vec {i}" for i in range(embeddings.shape[0])])

    # Add color bar
    cbar = fig.colorbar(cax, orientation="vertical", shrink=0.5)
    cbar.set_label("Embedding Value")

    # plt.title(f"")

    if show:
        plt.show()
    if out_path:
        print(f"Saving figure to {out_path}")
        plt.tight_layout()
        plt.savefig(out_path)


def load_embedding(file_path: Path) -> np.ndarray:
    """
    Load a SignCLIP embedding from a .npy file, ensuring it has the correct shape.

    Args:
        file_path (Path): Path to the .npy file.

    Returns:
        np.ndarray: The embedding with shape (768,).
    """
    embedding = np.load(file_path)
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding[0]  # Reduce shape from (1, 768) to (768,)
    return embedding


def get_embeddings_df_from_folder_of_embeddings(directory: Path):
    embedding_files = list(directory.rglob("*.npy"))

    # get the video ID (assuming VIDEOID-GLOSS-using-model-MODEL)
    embedding_files_df = pd.DataFrame(
        {
            "Embedding File Path": [str(embedding_file) for embedding_file in embedding_files],
            "Embedding File Name": [str(embedding_file.name) for embedding_file in embedding_files],
        }
    )

    # print(embedding_files_df.info())

    embedding_files_df["Embedding File Name"] = embedding_files_df["Embedding File Name"].astype(str)

    # Split by '-' and extract VIDEOID (first part)
    embedding_files_df["Video ID"] = embedding_files_df["Embedding File Name"].str.split("-").str[0]

    # Split by '-using-model-' and extract MODEL (second part)
    embedding_files_df["Embedding Model"] = (
        embedding_files_df["Embedding File Name"]
        .str.split("-using-model-")
        .str[-1]
        .str.replace(".npy", "", regex=False)
    )

    # print(embedding_files_df[["Video ID", "Embedding Model"]])
    return embedding_files_df


def visualize_embeddings_for_gloss(
    df,
    model,
    gloss,
    embedding_count: Optional[int] = None,
    dim_min: int = 0,
    dim_max: int = 768,
    show: bool = False,
    out_folder: Optional[Path] = None,
):

    df = df[df["Embedding Model"] == model]

    df = df[df["Gloss"] == gloss]

    # print(df.info())

    embeddings_list = [load_embedding(embed_path) for embed_path in df["Embedding File Path"].tolist()]
    embeddings = np.array(embeddings_list)
    print(embeddings.shape)
    if out_folder is not None:
        if embedding_count is None:
            embedding_count = len(embeddings_list)
        out_path = (
            out_folder / f"{gloss}_embeddings_count_{embedding_count}_mindim_{dim_min}_maxdim_{dim_max}_colorcoded.png"
        )
    else:
        out_path = None

    visualize_embeddings(
        embeddings,
        dim_min=dim_min,
        dim_max=dim_max,
        embedding_count=embedding_count,
        show=show,
        out_path=out_path,
        gloss=gloss,
    )


def get_regions(min_val, max_val, divisions):
    # Calculate the step size for the specified number of divisions
    step = (max_val - min_val) // divisions

    # Generate regions as tuples of start and end indices
    for i in range(divisions):
        start = min_val + i * step
        end = start + step
        yield (start, end)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Show color-coded embeddings for gloss")
    parser.add_argument("glosses", type=str, nargs="+", help="List of gloss values for analysis")
    parser.add_argument(
        "--model", type=str, default="sem-lex", choices=["asl-citizen", "asl-signs", "baseline_temporal", "asl-citizen"]
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--embedding_count", type=int)
    parser.add_argument("--dim_min", type=int, default=0)
    parser.add_argument("--dim_max", type=int, default=768)
    parser.add_argument("--divisions", type=int, default=1)
    parser.add_argument("--out_folder", type=Path)
    args = parser.parse_args()

    dataset_stats_json_path = Path(r"C:\Users\Colin\data\ASL_Citizen\ASL_Citizen\asl_citizen_dataset_stats.json")
    embeddings_folder = Path(r"C:\Users\Colin\data\ASL_Citizen\embeddings\embeddings")
    embeddings_df = get_embeddings_df_from_folder_of_embeddings(embeddings_folder)

    model = args.model
    glosses = args.glosses

    # df = pd.DataFrame(dataset_stats_json)
    df = pd.read_json(dataset_stats_json_path)
    df = df.merge(embeddings_df, on="Video ID", how="left")
    # print(df.head())

    # print(embeddings_df.head())
    # for gloss in glosses:
    #     print(f"Visualizing {gloss}")
    #     visualize_embeddings_for_gloss(
    #         df=df,
    #         model=model,
    #         gloss=gloss,
    #         embedding_count=args.embedding_count,
    #         dim_min=args.dim_min,
    #         dim_max=args.dim_max,
    #         show=args.show,
    #         out_folder=args.out_folder,
    #     )

    out_dir = args.out_folder / "_".join(glosses)
    out_dir.mkdir(exist_ok=True)
    # https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/word2vec.py

    for dim_min, dim_max in get_regions(args.dim_min, args.dim_max, divisions=args.divisions):

        visualize_embeddings_for_multiple_glosses(
            df,
            model=model,
            gloss_list=glosses,
            embedding_count=args.embedding_count,
            dim_min=dim_min,
            dim_max=dim_max,
            show=args.show,
            out_folder=args.out_folder / "_".join(glosses),
        )
