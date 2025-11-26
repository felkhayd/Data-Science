class SelectCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X[self.cols]

class SessionStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        seqs = X["seq_raw"]
        feats = []
        for seq in seqs:
            if not isinstance(seq, list):
                feats.append([0, 0, 0])
                continue
            n_actions = sum(1 for s in seq if not (isinstance(s, str) and s.startswith("t") and s[1:].isdigit()))
            n_twins = sum(1 for s in seq if isinstance(s, str) and s.startswith("t") and s[1:].isdigit())
            unique_ctrl = len(set(re.findall(r"\((.*?)\)", " ".join(seq))))
            feats.append([n_actions, n_twins, unique_ctrl])
        return np.array(feats)

class MiniLMEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column="seq_txt", model_path=None, batch_size=64):
        """
        column: nom de la colonne texte à encoder
        model_path: chemin local du modèle (ex: '/Users/.../all-MiniLM-L6-v2')
        batch_size: taille du batch pour l'encodage
        """
        
        
        self.column = column
        self.model_path = model_path or "all-MiniLM-L6-v2"
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y=None):
        if self.model is None:
            device = "mps" if torch.backends.mps.is_available() else (
                     "cuda" if torch.cuda.is_available() else "cpu")
            print(f"🧠 Chargement MiniLM sur {device} ...")
            self.model = SentenceTransformer(self.model_path, device=device)
        return self

    def transform(self, X):
        texts = X[self.column].astype(str).tolist()
        print(f"⚙️ Encodage MiniLM de {len(texts)} séquences...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings