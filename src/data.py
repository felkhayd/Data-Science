from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import TruncatedSVD
from .features import (
    SelectCols,
    SessionStats
)


features_union = FeatureUnion([
    # ---- TF-IDF sur seq_txt (1D string array) ----
    ("tfidf", Pipeline([
        ("sel_txt", SelectCols(["seq_txt"])),
        ("to_1d", FunctionTransformer(lambda df: df["seq_txt"].astype(str).tolist(), validate=False)),
        ("tfidf", TfidfVectorizer(
            min_df=3,
            ngram_range=(1,1),
            max_features=20000,
            sublinear_tf=True,
            strip_accents="unicode"
        )),
        ("svd", TruncatedSVD(
            n_components=200,      # 🔧 à ajuster selon la taille du corpus
            random_state=42
        ))
    ])),
    ("nav", Pipeline([
        ("sel_nav", SelectCols(["navigateur"])),
        ("to_2d", FunctionTransformer(lambda df: df.values, validate=False)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])),
    # ---- Stats agrégées (inchangé) ----
    ("agg", Pipeline([
        ("sel_raw", SelectCols(["seq_raw"])),
        ("stats", SessionStats()),
        ("to_df", FunctionTransformer(lambda a: pd.DataFrame(
            a, columns=["n_actions","n_twins","n_unique_ctrl"]), validate=False))
    ]))
], n_jobs=-1)