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

st.markdown("""
    <style>
    /* Fondo general y color de texto */
    .stApp {
        background-color: #141414;
        color: white;
    }
    
    /* Títulos y fuentes */
    h1, h2, h3 {
        color: #E50914 !important; /* Rojo Netflix */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    /* Botones estilo Netflix */
    .stButton>button {
        background-color: #E50914 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: bold !important;
    }
    
    /* Espaciado para Modelos 1 y 2 */
    .netflix-spacer {
        margin-top: 0px;
    }

    /* Ajustes para la Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    
    /* --- MANTENIMIENTO DE ESTRUCTURA COLUMNAS MATCH --- */
    [data-testid="stHorizontalBlock"] {
        gap: 20px !important;
    }

    /* Columnas 1 y 4: 20px más pequeñas mediante padding */
    [data-testid="column"]:nth-of-type(1), 
    [data-testid="column"]:nth-of-type(4) {
        flex: 1.1 1 0% !important;
        padding-left: 25px !important;
        padding-right: 25px !important;
    }
    
    /* Columnas 2 y 3: Más grandes */
    [data-testid="column"]:nth-of-type(2), 
    [data-testid="column"]:nth-of-type(3) {
        flex: 1 1 0% !important;
        padding-left: 15px !important;
        padding-right: 15px !important;
    }

    .rec-title { font-size: 0.9rem !important; font-weight: bold; color: white; }
    .rec-info { font-size: 0.8rem !important; color: #bbb; }
    
    /* Estilo para los inputs */
    .stSelectbox, .stMultiSelect, .stSlider {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


if "shown_titles" not in st.session_state: st.session_state.shown_titles = []
if "current_item" not in st.session_state: st.session_state.current_item = None
if "liked_genres" not in st.session_state: st.session_state.liked_genres = set()
if "last_match" not in st.session_state: st.session_state.last_match = None


st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.title("Menú")
page = st.sidebar.radio(
    "Navegación",
    ["Recomendación por gustos", "Recomendación por Usuario", "Recomendación por Match"]
)

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
        filtered = df[df["Tipo"] == content_type]
        if user_genres:
            pattern = "|".join(user_genres)
            filtered = filtered[filtered["Genero"].str.contains(pattern, case=False, na=False)]
        
        if content_type == "Pelicula":
            filtered = filtered[filtered["duration_min"] >= user_min_duration]
            
        recs = filtered.sort_values(by="Clasificacion", ascending=False).head(10)
        st.subheader("Títulos recomendados:")
        for _, row in recs.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1: st.image(row["link"], width=120)
            with col2:
                st.markdown(f"### {row['Titulo']}")
                st.write(f"**Tipo:** {row['Tipo']} | **Género:** {row['Genero']}")
                st.write(f"⭐ **Clasificación:** {row['Clasificacion']}")
            st.divider()

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
            for genre in g.split(","): genres_watched.add(genre.strip())
        
        preferred_type = watched_df["Tipo"].mode()[0]
        pattern = "|".join(genres_watched) if genres_watched else ""
        
        recommendations = df[
            (~df["Titulo"].isin(watched_titles)) & 
            (df["Tipo"] == preferred_type) & 
            (df["Genero"].str.contains(pattern, case=False, na=False) if pattern else True)
        ].sort_values(by="Clasificacion", ascending=False).head(10)

        for _, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1: st.image(row["link"], width=120)
            with col2:
                st.markdown(f"### {row['Titulo']}")
                st.write(f"**Tipo:** {row['Tipo']} | **Género:** {row['Genero']}")
                st.write(f"⭐ **Clasificación:** {row['Clasificacion']}")
            st.divider()

elif page == "Recomendación por Match":
    st.markdown("<h2 style='text-align: center;'>Recomendación por Match</h2>", unsafe_allow_html=True)

    def get_new_item():
        available = df[~df["Titulo"].isin(st.session_state.shown_titles)]
        return available.sample(1).iloc[0] if not available.empty else None

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
                st.session_state.shown_titles.append(item["Titulo"])
                st.session_state.current_item = get_new_item()
                st.rerun()
            if cv.button("💚", key="v_btn", use_container_width=True):
                st.session_state.last_match = item
                st.session_state.liked_genres = set([g.strip() for g in str(item["Genero"]).split(",")])
                st.session_state.shown_titles.append(item["Titulo"])
                st.session_state.current_item = get_new_item()
                st.rerun()

    exclude = [item["Titulo"]] if item is not None else [""]
    if st.session_state.last_match is not None: exclude.append(st.session_state.last_match["Titulo"])

    def render_rec_detailed(row):
        sc1, sc2 = st.columns([1.2, 2])
        with sc1: st.image(row["link"], use_container_width=True)
        with sc2:
            st.markdown(f"<p class='rec-title'>{row['Titulo']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='rec-info'>{row['Genero']}<br>⭐ {row['Clasificacion']} | 🗳️ {int(row['votes'])}</p>", unsafe_allow_html=True)
        st.divider()

    with c2:
        st.markdown("<h3 style='text-align: center;'>Populares</h3>", unsafe_allow_html=True)
        if st.session_state.liked_genres:
            pattern = "|".join(st.session_state.liked_genres)
            recs = df[(df["Genero"].str.contains(pattern, case=False, na=False)) & (~df["Titulo"].isin(exclude)) & (df["votes"] >= 350)].sort_values(by="Clasificacion", ascending=False).head(5)
            for _, r in recs.iterrows(): render_rec_detailed(r)

    with c3:
        st.markdown("<h3 style='text-align: center;'>Joyas Desconicidas</h3>", unsafe_allow_html=True)
        if st.session_state.liked_genres:
            pattern = "|".join(st.session_state.liked_genres)
            recs = df[(df["Genero"].str.contains(pattern, case=False, na=False)) & (~df["Titulo"].isin(exclude)) & (df["votes"] < 350)].sort_values(by="Clasificacion", ascending=False).head(5)
            for _, r in recs.iterrows(): render_rec_detailed(r)

    with c4:
        st.markdown("<h3 style='text-align: center;'>Match con</h3>", unsafe_allow_html=True)
        if st.session_state.last_match is not None:
            m = st.session_state.last_match
            st.image(m["link"], use_container_width=True)
            st.markdown(f"**{m['Titulo']}**")
            st.caption(f"{m['Genero']} | ⭐ {m['Clasificacion']}")