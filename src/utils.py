import pandas as pd 
import warnings
from IPython.display import display, Markdown

def read_ds(dataset_type):
    """
    Lit les données train ou test avec gestion des lignes à longueur variable
    """
    filename = f'{dataset_type}.csv'
    
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split(',')
            
            if dataset_type == 'train':
                # Train: user_id, browser, actions...
                row = {
                    'user_id': fields[0],
                    'browser': fields[1],
                    'actions': fields[2:]
                }
            else:  # test
                # Test: browser, actions...
                row = {
                    'browser': fields[0],
                    'actions': fields[1:]
                }
            
            data.append(row)
    
    return pd.DataFrame(data)

# décorateurs utilitaires pour supprimer les avertissements de la sortie et imprimer un cadre de données dans un tableau Markdown.
def ignore_warnings(f):
    def _f(*args, **kwargs):
        warnings.filterwarnings('ignore')
        v = f(*args, **kwargs)
        warnings.filterwarnings('default')
        return v
    return _f

# affiche un DataFrame Pandas sous forme de tableau Markdown dans un notebook Jupyter.
def markdown_table(headNtail=False, use_index=True, title=None, precision=2):
    def _get_value(val): return str(round(val, precision) if isinstance(val, float) else val)
    def _format_row(row): 
        row_str = ""
        if use_index: row_str += f"|{str(row.name)}"
        for value in row.values: row_str += f"| {_get_value(value)}"
        return row_str + "|"
    def _get_str(df):
        return "\n".join(df.apply(_format_row, axis=1))
    def _deco(f):
        def _f(*args, **kwargs):
            df = f(*args, **kwargs)
            _str = f"#### {title}\n" if title else ""
            header = ([str(df.index.name)] if use_index else []) + df.columns.astype(str).to_list() 
            _str += f"|{'|'.join(header)}|" + f"\n|{'--|'*len(header)}\n" if header else None
            if headNtail:
                _str += _get_str(df.head())
                _str += "\n|...|...|\n"
                _str += _get_str(df.tail())
            else:
                _str += _get_str(df)
            display(Markdown(_str))
        return _f
    return _deco

# fonction utilitaire permettant d'obtenir une grille graphique à partir d'un nombre arbitraire de lignes/colonnes ou de données.
def get_grid(n, n_row=None, n_col=None, titles=None, figsize=(10, 8), wspace=.5, hspace=.5, **kwargs):
    if n_row: n_col= n_col or math.floor(n/n_row)
    elif n_col: n_row= n_row or math.ceil(n/n_col)
    else:
        n_row = math.ceil(math.sqrt(n))
        n_col = math.floor(n/n_row)
    fig, axs = plt.subplots(n_row, n_col, figsize=figsize, **kwargs)
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    if titles is not None:
        for ax, title in zip(axs.flat, titles): ax.set_title(title)
    return fig, axs

def normalize_token(tok: str) -> list:
    tok = tok.strip()
    if not tok: return []
    if tok.startswith("t") and tok[1:].isdigit():
        return [f"TWIN_{tok[1:]}"]  # fenêtre tXX -> token temporel

    m = ACTION_RE.match(tok)
    if not m:
        # ex. "Double-clic" simple, ou token inconnu
        return [tok.replace(" ", "_")]
    base = m.group("base").strip().replace(" ", "_")
    ctrl = (m.group("ctrl") or "").strip().replace(".", "_").replace(" ", "_")
    conf = (m.group("conf") or "").strip().replace(" ", "_")
    chain = (m.group("chain") or "").strip().replace(" ", "_")
    edit = m.group("edit")

    out = [f"A_{base}"]
    if ctrl:  out.append(f"C_{ctrl}")
    if conf:  out.append(f"CFG_{conf}")
    if chain: out.append(f"CH_{chain}")
    if edit:  out.append("EDIT_1")
    return out

def seq_to_text(seq_list):
    toks = []
    for tok in seq_list:
        toks.extend(normalize_token(tok))
    # on garde aussi quelques bigrams implicites: TF-IDF n-grammes gérera la cooccurrence
    return " ".join(toks)

def row_to_sequence(row, start_col=2):
    vals = []
    for c in row.index[start_col:]:
        v = row[c]
        if pd.isna(v): break
        vals.append(str(v))
    return vals