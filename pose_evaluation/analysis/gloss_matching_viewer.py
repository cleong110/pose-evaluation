import random
import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt

# Optional: for visual overlap
import pandas as pd
import streamlit as st
from num2words import num2words
from supervenn import supervenn


def convert_digits_to_words(s: str) -> str:
    # If the whole string is a digit, convert it
    if s.isdigit():
        return num2words(int(s)).upper()

    # If string ends with a digit and has preceding non-digit characters, skip
    if re.match(r"^.+\d$", s):
        return s

    # Replace digits within the string (not trailing ones with letters before) with words
    def replace_match(match):
        return num2words(int(match.group())).upper()

    return re.sub(r"\d+", replace_match, s)


def create_gloss_tuple(row, col1: str, col2: str):
    gloss_1 = row[col1].upper()
    gloss_2 = row[col2].upper()
    return tuple(sorted([gloss_1, gloss_2]))


def load_csv(path: Path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        with open(path, "rb") as f:
            raw = f.read()
        decoded = raw.decode("utf-8", errors="replace")
        return pd.read_csv(StringIO(decoded), **kwargs)


def find_weird(df: pd.DataFrame, cols: list) -> set:
    vocab = []
    for col in cols:
        vocab.extend(df[col].dropna().unique())
    vocab = list(set(vocab))

    weird_chars = set()
    for s in vocab:
        non_alnum_chars = {c for c in s if not c.isalnum() and c != "_"}
        if non_alnum_chars:
            weird_chars.update(non_alnum_chars)
    return weird_chars


def merge_and_find_unmatched(
    df,  # the dataset df, Sem-Lex or ASL Citizen or ASLKG
    asl_lex_df,  # the ASL Lex data, loaded from signdata.csv
    left_on="ASL-LEX Code",
    keep_left: list | None = None,
    right_on="Code",
    keep_right: list | None = None,
):
    # df[left_on] = df[left_on].str.upper()
    # asl_lex_df[right_on] = asl_lex_df[right_on].str.upper()
    df = df.copy()
    asl_lex_df = asl_lex_df.copy()

    if keep_left is None:
        keep_left = left_on

    if keep_right is None:
        keep_right = list({"EntryID", right_on})

    df = df[keep_left]
    merged = df.merge(
        asl_lex_df[keep_right],
        left_on=left_on,
        right_on=right_on,
        how="outer",
        indicator=True,
    )
    left_only = merged[merged["_merge"] == "left_only"]
    right_only = merged[merged["_merge"] == "right_only"]
    return merged, left_only, right_only


def build_supervenn_sets(df_dict):
    sets = {}
    for name, df in df_dict.items():
        columns = df.columns.tolist()
        col = st.selectbox(
            f"What Column to use for {name}?",
            options=columns,
            index=columns.index("EntryID") if "EntryID" in columns else 0,
        )
        # glosses = set(df["GLOSS_A"].unique()).union(df["GLOSS_B"].unique())
        glosses = set(df[col])
        sets[name] = glosses
    return sets


def select_merge_columns(
    df: pd.DataFrame,
    df_name: str,
    default_merge_col: str,
    key_prefix: str,
    side: str,
) -> tuple[list[str], str, pd.DataFrame]:
    """Interactive column selector and filterer for one side of a merge."""
    df_columns = df.columns.tolist()

    merge_on = st.selectbox(
        f"{df_name} column for merge",
        df_columns,
        index=df_columns.index(default_merge_col) if default_merge_col in df_columns else 0,
        key=f"{key_prefix}_select_{side}_on",
    )
    df[f"{merge_on} original"] = df[merge_on]

    keep_cols = [f"{merge_on} original", merge_on]
    keep_cols.extend(
        st.multiselect(
            f"Additional Columns to keep for {df_name}",
            options=[col for col in df_columns if col != merge_on],
            default=[col for col in ["EntryID", "EntryID original"] if col in df_columns and col != merge_on],
            key=f"{key_prefix}_select_keep_{side}",
        )
    )

    for col in keep_cols:
        st.write(f"{df_name} {col} has {df[col].nunique()} unique values")

    drop_na = st.checkbox(
        f"Drop rows of {df_name} where any of {keep_cols} is None/NaN",
        key=f"{key_prefix}_dropna_{side}",
        value=True,
    )

    filtered_df = df.dropna(subset=keep_cols) if drop_na else df.copy()

    apply_upper = st.checkbox(
        f"Apply uppercasing to {merge_on}",
        key=f"{key_prefix}_applyupper_{side}",
    )

    if apply_upper:
        filtered_df[merge_on] = filtered_df[merge_on].str.upper()

    apply_num2words = st.checkbox(
        f"Apply num2words to {merge_on}",
        key=f"{key_prefix}_apply_num2words_{side}",
    )
    if apply_num2words:
        filtered_df[merge_on] = filtered_df[merge_on].astype(str).apply(convert_digits_to_words)

    if st.checkbox(
        f"Show preview of filtered {df_name} ({len(df) - len(filtered_df)} rows dropped)",
        key=f"{key_prefix}_preview_{side}",
    ):
        st.dataframe(filtered_df[keep_cols])

    return keep_cols, merge_on, filtered_df


def interactive_merge(
    left_df: pd.DataFrame,
    left_df_name: str,
    right_df: pd.DataFrame,
    right_df_name: str,
):
    left_df = left_df.copy()
    right_df = right_df.copy()

    st.subheader(f"Merge Check: {left_df_name} and {right_df_name}")
    col1, col2 = st.columns(2)

    with col1:
        keep_left, left_on, left_df = select_merge_columns(
            left_df,
            left_df_name,
            default_merge_col="ASL-LEX Code",
            key_prefix=f"{left_df_name}_and_{right_df_name}",
            side="left",
        )

    with col2:
        keep_right, right_on, right_df = select_merge_columns(
            right_df,
            right_df_name,
            default_merge_col="Code",
            key_prefix=f"{left_df_name}_and_{right_df_name}",
            side="right",
        )

    merged_df, left_only, right_only = merge_and_find_unmatched(
        left_df,
        right_df,
        left_on=left_on,
        right_on=right_on,
        keep_left=keep_left,
        keep_right=keep_right,
    )

    st.write(f"{len(left_only)} Unmatched in {left_df_name}:")
    st.dataframe(left_only)

    st.write(f"{len(right_only)} Unmatched in {right_df_name}")
    st.dataframe(right_only)

    st.write(f"{len(merged_df)} Matches")
    merged_df = merged_df[merged_df["_merge"] == "both"]
    for col in merged_df.columns:
        col_samples = list(merged_df[col].unique())
        random.shuffle(col_samples)
        st.write(f"{col} has {len(merged_df[col].unique())} unique values, a sample is: {col_samples[:5]}")

    merged_sample = merged_df.sample(10)
    if st.button("Resample", key=f"{left_df_name}_and_{right_df_name}_show_sample"):
        merged_sample = merged_df.sample(10)
    st.dataframe(merged_sample)

    return merged_df


# Streamlit App
st.title("Gloss Pair Explorer")

DATA_PATH = Path("/opt/home/cleong/projects/semantic_and_visual_similarity/local_data")

aslkg_csv = DATA_PATH / "ASLKG/edges_v2_noweights.tsv"
sem_lex_csv = DATA_PATH / "Sem-Lex/semlex_metadata.csv"
asl_citizen_test_csv = DATA_PATH / "ASL_Citizen/splits/test.csv"
asl_lex_2_csv = DATA_PATH / "ASLLEX/signdata.csv"
lookalikes_csv = DATA_PATH / "SimilarSigns/deduped_sorted_similar_gloss_pairs.csv"

st.sidebar.header("Data Selection")
show_weird_chars = st.sidebar.checkbox("Show Weird Characters", value=True)
show_supervenn = st.sidebar.checkbox("Show Supervenn Overlap", value=False)


# ASL Knowledge Graph
aslkg_df = load_csv(aslkg_csv, delimiter="\t")
aslkg_df = aslkg_df[aslkg_df["relation"] == "response"]
aslkg_df = aslkg_df[aslkg_df["subject"].str.contains("asllex:") & aslkg_df["object"].str.contains("asllex:")]
aslkg_df["subject Original"] = aslkg_df["subject"]
aslkg_df["subject"] = aslkg_df["subject"].str.replace("asllex:", "").str.replace("#", "")
aslkg_df["object Original"] = aslkg_df["subject"]
aslkg_df["object"] = aslkg_df["object"].str.replace("asllex:", "").str.replace("#", "")
aslkg_df["gloss_tuple"] = aslkg_df.apply(create_gloss_tuple, col1="subject", col2="object", axis=1)
aslkg_df["relation"] = "Semantic"
aslkg_df = aslkg_df.rename(columns={"subject": "GLOSS_A", "object": "GLOSS_B"})

sem_lex_df = load_csv(sem_lex_csv)
sem_lex_df = sem_lex_df[sem_lex_df["label_type"] == "asllex"]

asl_citizen_df = load_csv(asl_citizen_test_csv)
lookalikes_df = load_csv(lookalikes_csv)
lookalikes_df["gloss_tuple"] = lookalikes_df.apply(create_gloss_tuple, col1="GLOSS_A", col2="GLOSS_B", axis=1)
lookalikes_df["relation"] = "Lookalike"

asl_lex_df = load_csv(asl_lex_2_csv)

if show_weird_chars:
    st.subheader("Unusual Characters in Labels")
    for name, (df, cols) in {
        "ASLKG": (aslkg_df, ["GLOSS_A", "GLOSS_B"]),
        "Sem-Lex": (sem_lex_df, ["label"]),
        "ASL Citizen": (asl_citizen_df, ["Gloss"]),
        "ASL Lex 2.0": (asl_lex_df, ["EntryID"]),
    }.items():
        chars = find_weird(df, cols)
        if chars:
            st.write(f"{name}: {chars}")

# Merge diagnostics
asl_citizen_merged_df = interactive_merge(asl_citizen_df, "ASL Citizen", asl_lex_df, "ASL Lex 2.0")


#
semlex_merged_df = interactive_merge(sem_lex_df, "Sem-Lex", asl_lex_df, "ASL Lex 2.0")

# Combine semantic and lookalike
combined_df = pd.concat([aslkg_df, lookalikes_df], ignore_index=True)
relation_df = (
    combined_df.groupby("gloss_tuple")["relation"]
    .agg(lambda x: "Both" if len(set(x)) > 1 else next(iter(x)))
    .reset_index()
)
gloss_info_df = combined_df.drop_duplicates(subset="gloss_tuple")[["gloss_tuple", "GLOSS_A", "GLOSS_B"]]
final_relations_df = gloss_info_df.merge(relation_df, on="gloss_tuple")
st.subheader("Final Gloss Pair Data")
st.write(final_relations_df)


for relation_type in final_relations_df["relation"].unique():
    relation_df = final_relations_df[final_relations_df["relation"] == relation_type]
    st.write(f"Relation Type: {relation_type}: {len(relation_df)} entries")
    st.dataframe(relation_df)

csv_output = st.download_button(
    label="Download Final Gloss Pairs as CSV",
    data=final_relations_df[["GLOSS_A", "GLOSS_B", "relation"]].to_csv(index=False),
    file_name="combined_gloss_pairs.csv",
    mime="text/csv",
)


if show_supervenn:
    st.subheader("Supervenn Diagram")

    df_sets = build_supervenn_sets(
        {
            "ASL Citizen (merged with ASL Lex 2.0)": asl_citizen_merged_df,
            "Sem-Lex (merged with ASL LEX 2.0)": semlex_merged_df,
            "ASL-Lex 2.0": asl_lex_df,
        }
    )

    sets = []
    labels = []
    for name, df_set in df_sets.items():
        # st.write(name)
        labels.append(name)
        sets.append(df_set)
        # st.write(len(df_set))
    fig, ax = plt.subplots(figsize=(12, 6))

    supervenn(sets, labels, ax=ax)
    st.write("Plotting Supervenn...")
    # st.write(fig)
    st.pyplot(fig)
