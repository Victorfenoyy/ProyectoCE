import streamlit as st
import pandas as pd
import os

st.set_page_config(layout="wide", page_title="Netflix Recommender")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "movies_with_images.csv")
df = pd.read_csv(csv_path)

df['votes'] = df['votes'].astype(str).str.replace(r'[,\.\s]', '', regex=True)
df['votes'] = pd.to_numeric(df['votes'], errors='coerce')
df = df.dropna(subset=['votes'])
df['Clasificacion'] = pd.to_numeric(df['Clasificacion'], errors='coerce').fillna(0)
df["Genero"] = df["Genero"].fillna("").astype(str)
df["duration_min"] = df["Duracion"].str.replace(" min", "", regex=False).astype(float)
df["Tipo"] = df["Tipo"].str.strip()

# =========================
# Modelo de IA (entrenado) para recomendaciones
# =========================
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

try:
    from scipy.sparse import hstack, csr_matrix
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


@st.cache_resource(show_spinner=False)
def _train_recommender_model(df_in: pd.DataFrame):
    text = (
        df_in["Titulo"].fillna("").astype(str) + " " +
        df_in["Genero"].fillna("").astype(str) + " " +
        df_in["Tipo"].fillna("").astype(str)
    ).str.lower()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=8000
    )
    X_text = vectorizer.fit_transform(text)

    num_cols = ["Clasificacion", "votes", "duration_min"]
    X_num = df_in[num_cols].fillna(0).astype(float).values
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    if _HAS_SCIPY:
        X = hstack([X_text, csr_matrix(X_num_scaled)])
    else:
        X = np.hstack([X_text.toarray(), X_num_scaled])

    n_components = 120
    n_components = int(min(n_components, max(2, min(X_text.shape[0]-1, X_text.shape[1]-1))))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_latent = svd.fit_transform(X)

    # Normalización L2 (para coseno)
    norms = np.linalg.norm(X_latent, axis=1, keepdims=True) + 1e-9
    X_latent = X_latent / norms

    return vectorizer, scaler, svd, X_latent


@st.cache_resource(show_spinner=False)
def _build_genre_matrix(df_in: pd.DataFrame):
    """
    Construye una matriz binaria de géneros (MultiLabelBinarizer).
    """
    def _split_genres(s):
        if pd.isna(s) or not str(s).strip():
            return []
        return [g.strip().lower() for g in str(s).split(",") if g.strip()]

    mlb = MultiLabelBinarizer()
    genre_lists = df_in["Genero"].apply(_split_genres).tolist()
    G = mlb.fit_transform(genre_lists).astype(np.int8)  # (n_items, n_genres)
    return mlb, G


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(vec) + 1e-9)
    return vec / n


def _embed_query(genres, content_type, vectorizer, scaler, svd) -> np.ndarray:
    q_text = (" ".join([str(g) for g in genres]) + " " + str(content_type)).strip().lower()

    q_text_vec = vectorizer.transform([q_text])

    q_num = scaler.mean_.reshape(1, -1)
    q_num_scaled = scaler.transform(q_num)

    if _HAS_SCIPY:
        q_vec = hstack([q_text_vec, csr_matrix(q_num_scaled)])
    else:
        q_vec = np.hstack([q_text_vec.toarray(), q_num_scaled])

    q_lat = svd.transform(q_vec)[0]
    return _l2_normalize(q_lat)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn = np.min(x)
    mx = np.max(x)
    den = (mx - mn) if (mx - mn) != 0 else 1.0
    return (x - mn) / den


def _recommend_genre_priority(
    df_in: pd.DataFrame,
    item_embeddings: np.ndarray,
    user_vec: np.ndarray,
    mlb: MultiLabelBinarizer,
    G: np.ndarray,
    query_genres,
    final_n: int,
    pool_size: int = 800,
    exclude_titles=None,
    filter_mask=None,
    w_genre: float = 0.30,
    w_sim: float = 0.60,
    w_rating: float = 0.05,
    w_votes: float = 0.05,
) -> pd.DataFrame:

    if exclude_titles is None:
        exclude_titles = set()

    # 1) Similitud coseno (porque embeddings están normalizados)
    sims_all = item_embeddings @ user_vec

    # 2) Máscara válida (solo para excluir títulos vistos / mostrados)
    valid = np.ones(len(df_in), dtype=bool)
    if exclude_titles:
        valid &= ~df_in["Titulo"].isin(list(exclude_titles)).values
    if filter_mask is not None:
        valid &= filter_mask.values if isinstance(filter_mask, pd.Series) else np.asarray(filter_mask, dtype=bool)

    sims_masked = sims_all.copy()
    sims_masked[~valid] = -1e9

    # 3) Pool TOP-K por similitud (acelera y mantiene relevancia)
    k = int(min(pool_size, int(np.sum(valid)))) if np.sum(valid) > 0 else 0
    if k <= 0:
        return df_in.head(0)

    pool_idx = np.argsort(-sims_masked)[:k]
    pool = df_in.iloc[pool_idx].copy()

    # 4) Score de géneros
    q = [str(g).strip().lower() for g in (query_genres or []) if str(g).strip()]
    q_vec_bin = np.zeros(len(mlb.classes_), dtype=np.int8)
    class_to_i = {c: i for i, c in enumerate(mlb.classes_)}

    for g in q:
        if g in class_to_i:
            q_vec_bin[class_to_i[g]] = 1

    q_len = int(q_vec_bin.sum())

    if q_len == 0:
        genre_score = np.zeros(len(pool_idx), dtype=float)
    else:
        overlap = (G[pool_idx] @ q_vec_bin).astype(float)

        if q_len == 1:
            genre_score = (overlap > 0).astype(float)
        else:
            coverage = overlap / q_len
            exact = (overlap == q_len).astype(float)
            genre_score = coverage + 0.25 * exact

    # 5) Normalizaciones (para mezclar señales)
    pool_sim = _minmax(sims_all[pool_idx])
    pool_rating = _minmax(pool["Clasificacion"].fillna(0).astype(float).values)
    pool_votes = _minmax(np.log1p(pool["votes"].fillna(0).astype(float).values))

    # 6) Score final (selección TOP relevante)
    score = (
        w_genre * genre_score +
        w_sim * pool_sim +
        w_rating * pool_rating +
        w_votes * pool_votes
    )

    pool["_score"] = score

    pool = pool.sort_values(by=["_score"], ascending=False).head(final_n).copy()

    # Orden final (lo que se muestra)
    pool = pool.sort_values(by=["Clasificacion"], ascending=False)

    return pool.drop(columns=["_score"])


# Entrenamos modelo y matriz de géneros (cacheado)
_vectorizer, _scaler, _svd, _ITEM_EMB = _train_recommender_model(df)
_mlb, _G = _build_genre_matrix(df)


st.markdown("""
    <style>
    .stApp {
        background-color: #141414;
        color: white;
    }
    h1, h2, h3 {
        color: #E50914 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton>button {
        background-color: #E50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: bold !important;
    }
    .netflix-spacer {
        margin-top: 0px;
    }
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    [data-testid="stHorizontalBlock"] {
        gap: 20px !important;
    }
    [data-testid="column"]:nth-of-type(1), 
    [data-testid="column"]:nth-of-type(4) {
        flex: 1.1 1 0% !important;
        padding-left: 25px !important;
        padding-right: 25px !important;
    }
    [data-testid="column"]:nth-of-type(2), 
    [data-testid="column"]:nth-of-type(3) {
        flex: 1 1 0% !important;
        padding-left: 15px !important;
        padding-right: 15px !important;
    }
    .rec-title { font-size: 0.9rem !important; font-weight: bold; color: white; }
    .rec-info { font-size: 0.8rem !important; color: #bbb; }
    .stSelectbox, .stMultiSelect, .stSlider {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


if "shown_titles" not in st.session_state: st.session_state.shown_titles = []
if "current_item" not in st.session_state: st.session_state.current_item = None
if "liked_genres" not in st.session_state: st.session_state.liked_genres = set()
if "last_match" not in st.session_state: st.session_state.last_match = None

if "swipe_count" not in st.session_state: st.session_state.swipe_count = 0
if "liked_titles" not in st.session_state: st.session_state.liked_titles = []


st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.title("Menú")
page = st.sidebar.radio(
    "Navegación",
    ["Recomendación por gustos", "Recomendación por Usuario", "Recomendación por Match"]
)

# =========================
# MODELO 1: Recomendación por gustos (SIN FILTROS)
# =========================
if page == "Recomendación por gustos":
    st.markdown('<div class="netflix-spacer"></div>', unsafe_allow_html=True)
    st.title("Sistema de Recomendación por Gustos")

    content_type = st.selectbox("¿Qué quieres ver?", ["Pelicula", "Serie"])

    all_genres = set()
    for g in df["Genero"].dropna():
        for genre in g.split(","):
            all_genres.add(genre.strip())

    user_genres = st.multiselect("¿Qué géneros te gustan?", sorted(all_genres))
    user_min_duration = st.slider("Duración mínima (minutos)", 30, 240, 50)

    if st.button("Get recommendations"):
        # Antes: filtrabas por tipo/género/duración.
        # Ahora: NO recortamos el dataset -> la IA rankea sobre todo el catálogo.
        q_vec = _embed_query(user_genres, content_type, _vectorizer, _scaler, _svd)

        recs = _recommend_genre_priority(
            df,
            _ITEM_EMB,
            q_vec,
            _mlb,
            _G,
            user_genres,
            final_n=10,
            pool_size=800
        )

        st.subheader("Títulos recomendados:")
        for _, row in recs.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1: st.image(row["link"], width=120)
            with col2:
                st.markdown(f"### {row['Titulo']}")
                st.write(f"**Tipo:** {row['Tipo']} | **Género:** {row['Genero']}")
                st.write(f"⭐ **Clasificación:** {row['Clasificacion']}")
            st.divider()

# =========================
# MODELO 2: Recomendación por Usuario (SIN FILTROS DE TIPO/GÉNERO)
# =========================
elif page == "Recomendación por Usuario":
    st.markdown('<div class="netflix-spacer"></div>', unsafe_allow_html=True)
    st.title("Recomendación basada en tu historial")

    username = st.text_input("Introduce tu nombre de usuario")
    watched_titles = st.multiselect("Selecciona lo que ya has visto", options=sorted(df["Titulo"].unique()))

    if username and watched_titles:
        st.success(f"{username}, analizando tu historial...")
        watched_df = df[df["Titulo"].isin(watched_titles)]

        genres_watched = set()
        for g in watched_df["Genero"].dropna():
            for genre in g.split(","):
                genres_watched.add(genre.strip())

        preferred_type = watched_df["Tipo"].mode()[0] if not watched_df.empty else "Pelicula Serie"

        watched_idx = df[df["Titulo"].isin(watched_titles)].index
        if len(watched_idx) > 0:
            user_vec = _ITEM_EMB[watched_idx].mean(axis=0)
            user_vec = _l2_normalize(user_vec)
        else:
            user_vec = _embed_query(genres_watched, preferred_type, _vectorizer, _scaler, _svd)

        # Antes: candidate_mask filtraba por tipo y por géneros.
        # Ahora: NO filtramos -> solo excluimos lo ya visto.
        recommendations = _recommend_genre_priority(
            df,
            _ITEM_EMB,
            user_vec,
            _mlb,
            _G,
            list(genres_watched),
            final_n=10,
            pool_size=800,
            exclude_titles=set(watched_titles),
            filter_mask=None
        )

        for _, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1: st.image(row["link"], width=120)
            with col2:
                st.markdown(f"### {row['Titulo']}")
                st.write(f"**Tipo:** {row['Tipo']} | **Género:** {row['Genero']}")
                st.write(f"⭐ **Clasificación:** {row['Clasificacion']}")
            st.divider()

# =========================
# MODELO 3: Recomendación por Match
# =========================
elif page == "Recomendación por Match":
    st.markdown("<h2 style='text-align: center;'>Recomendación por Match</h2>", unsafe_allow_html=True)

    # A partir de 5 swipes, elegir siguiente item por similitud a los LIKES
    def get_new_item():
        available = df[~df["Titulo"].isin(st.session_state.shown_titles)]
        if available.empty:
            return None

        # Primeras 5: exploración aleatoria
        if st.session_state.swipe_count < 5:
            return available.sample(1).iloc[0]

        # A partir de 5: si hay likes, perfil con likes
        if len(st.session_state.liked_titles) > 0:
            like_titles = st.session_state.liked_titles[-10:]  # últimos 10 likes
            idxs = []
            for t in like_titles:
                found = df.index[df["Titulo"] == t]
                if len(found) > 0:
                    idxs.append(found[0])

            if len(idxs) > 0:
                profile_vec = _ITEM_EMB[idxs].mean(axis=0)
                profile_vec = _l2_normalize(profile_vec)

                cand_idx = available.index.values
                sims = _ITEM_EMB[cand_idx] @ profile_vec

                best_local = int(np.argmax(sims))
                best_idx = int(cand_idx[best_local])
                return df.loc[best_idx]

        return available.sample(1).iloc[0]

    if st.session_state.current_item is None:
        st.session_state.current_item = get_new_item()

    item = st.session_state.current_item
    c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1.1])

    with c1:
        st.markdown("<h3 style='text-align: center;'></h3>", unsafe_allow_html=True)
        if item is not None:
            st.image(item["link"], use_container_width=True)
            st.markdown(f"**{item['Titulo']}**")
            st.caption(f"🎭 {item['Genero']} | ⭐ {item['Clasificacion']}")
            cx, cv = st.columns(2)

            if cx.button("❌", key="x_btn", use_container_width=True):
                st.session_state.swipe_count += 1
                st.session_state.shown_titles.append(item["Titulo"])
                st.session_state.current_item = get_new_item()
                st.rerun()

            if cv.button("💚", key="v_btn", use_container_width=True):
                st.session_state.swipe_count += 1
                st.session_state.liked_titles.append(item["Titulo"])

                st.session_state.last_match = item
                st.session_state.liked_genres = set([g.strip() for g in str(item["Genero"]).split(",")])
                st.session_state.shown_titles.append(item["Titulo"])
                st.session_state.current_item = get_new_item()
                st.rerun()

    exclude = [item["Titulo"]] if item is not None else [""]
    if st.session_state.last_match is not None:
        exclude.append(st.session_state.last_match["Titulo"])

    def render_rec_detailed(row):
        sc1, sc2 = st.columns([1.2, 2])
        with sc1: st.image(row["link"], use_container_width=True)
        with sc2:
            st.markdown(f"<p class='rec-title'>{row['Titulo']}</p>", unsafe_allow_html=True)
            st.markdown(
                f"<p class='rec-info'>{row['Genero']}<br>⭐ {row['Clasificacion']} | 🗳️ {int(row['votes'])}</p>",
                unsafe_allow_html=True
            )
        st.divider()

    with c2:
        st.markdown("<h3 style='text-align: center;'>Populares</h3>", unsafe_allow_html=True)
        if st.session_state.liked_titles:
            # Perfil a partir de likes (más “IA”)
            idxs = []
            for t in st.session_state.liked_titles[-10:]:
                found = df.index[df["Titulo"] == t]
                if len(found) > 0:
                    idxs.append(found[0])

            base_vec = _l2_normalize(_ITEM_EMB[idxs].mean(axis=0)) if idxs else _embed_query(
                st.session_state.liked_genres, "Pelicula Serie", _vectorizer, _scaler, _svd
            )

            # Mantengo solo la idea de "popular" por votos (si quieres quitarlo, dímelo)
            cand_mask = (df["votes"] >= 350) & (~df["Titulo"].isin(exclude))

            recs = _recommend_genre_priority(
                df,
                _ITEM_EMB,
                base_vec,
                _mlb,
                _G,
                list(st.session_state.liked_genres),
                final_n=5,
                pool_size=800,
                exclude_titles=set(exclude),
                filter_mask=cand_mask
            )

            for _, r in recs.iterrows():
                render_rec_detailed(r)

    with c3:
        st.markdown("<h3 style='text-align: center;'>Joyas Desconicidas</h3>", unsafe_allow_html=True)
        if st.session_state.liked_titles:
            idxs = []
            for t in st.session_state.liked_titles[-10:]:
                found = df.index[df["Titulo"] == t]
                if len(found) > 0:
                    idxs.append(found[0])

            base_vec = _l2_normalize(_ITEM_EMB[idxs].mean(axis=0)) if idxs else _embed_query(
                st.session_state.liked_genres, "Pelicula Serie", _vectorizer, _scaler, _svd
            )

            # Mantengo solo la idea de "joya" por votos (si quieres quitarlo, dímelo)
            cand_mask = (df["votes"] < 350) & (~df["Titulo"].isin(exclude))

            recs = _recommend_genre_priority(
                df,
                _ITEM_EMB,
                base_vec,
                _mlb,
                _G,
                list(st.session_state.liked_genres),
                final_n=5,
                pool_size=800,
                exclude_titles=set(exclude),
                filter_mask=cand_mask
            )

            for _, r in recs.iterrows():
                render_rec_detailed(r)

    with c4:
        st.markdown("<h3 style='text-align: center;'>Match con</h3>", unsafe_allow_html=True)
        if st.session_state.last_match is not None:
            m = st.session_state.last_match
            st.image(m["link"], use_container_width=True)
            st.markdown(f"**{m['Titulo']}**")
            st.caption(f"{m['Genero']} | ⭐ {m['Clasificacion']}")
