import streamlit as st
import pandas as pd
import os
import random


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "movies_with_images.csv")
df = pd.read_csv(csv_path)
df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0)

# Opcional: Asegurarnos que Clasificacion también sea numérica
df['Clasificacion'] = pd.to_numeric(df['Clasificacion'], errors='coerce').fillna(0)

df["Genero"] = df["Genero"].fillna("").astype(str)

# --------------------------------------------------
# Preprocess
# --------------------------------------------------
df["duration_min"] = (
    df["Duracion"]
    .str.replace(" min", "", regex=False)
    .astype(float)
)

df["Tipo"] = df["Tipo"].str.strip()

# --------------------------------------------------
# Session state (Tinder mode)
# --------------------------------------------------
if "shown_titles" not in st.session_state:
    st.session_state.shown_titles = []

if "current_item" not in st.session_state:
    st.session_state.current_item = None

if "used_genres" not in st.session_state:
    st.session_state.used_genres = set()

# --------------------------------------------------
# Sidebar menu
# --------------------------------------------------
st.sidebar.title("📺 Menú")
page = st.sidebar.radio(
    "Selecciona una opción",
    [
        "Recomendación por gustos",
        "Recomendación por Usuario",
        "Recomendación por match"
    ]
)

# ==================================================
# PAGE 1: Recomendación por gustos
# ==================================================
if page == "Recomendación por gustos":

    st.title("Sistema de Recomendación por Gustos")

    content_type = st.selectbox(
        "¿Qué quieres ver?",
        ["Pelicula", "Serie"]
    )

    all_genres = set()
    for g in df["Genero"].dropna():
        for genre in g.split(","):
            all_genres.add(genre.strip())

    user_genres = st.multiselect(
        "¿Qué géneros te gustan?",
        sorted(all_genres)
    )

    user_min_duration = st.slider(
        "Duración mínima (minutos)",
        min_value=30,
        max_value=240,
        value=50
    )

    def recommend_by_preferences(df, content_type, genres, min_minutes):
        filtered = df[df["Tipo"] == content_type]

        if genres:
            pattern = "|".join(genres)
            filtered = filtered[
                filtered["Genero"].str.contains(pattern, case=False, na=False)
            ]

        if content_type == "Pelicula":
            filtered = filtered[filtered["duration_min"] >= min_minutes]

        return filtered.sort_values(
            by="Clasificacion", ascending=False
        ).head(10)

    if st.button("Get recommendations"):
        recs = recommend_by_preferences(
            df, content_type, user_genres, user_min_duration
        )

        st.subheader("Títulos recomendados:")

        for _, row in recs.iterrows():
            col1, col2 = st.columns([1, 3])

            with col1:
                st.image(row["link"], width=120)

            with col2:
                st.markdown(f"### {row['Titulo']}")
                st.write(f"**Tipo:** {row['Tipo']}")
                st.write(f"**Género:** {row['Genero']}")
                st.write(f"**Duración:** {row['Duracion']}")
                st.write(f"⭐ **Clasificación:** {row['Clasificacion']}")

            st.divider()

# ==================================================
# PAGE 2: RECOMMENDATION BY USER HISTORY
# ==================================================
elif page == "Recomendación por Usuario":

    st.title("Recomendación basada en tu historial")

    username = st.text_input("Introduce tu nombre de usuario")

    watched_titles = st.multiselect(
        "Selecciona las películas o series que ya has visto",
        options=sorted(df["Titulo"].unique())
    )

    if username and watched_titles:

        st.success(f"{username}, has visto {len(watched_titles)} títulos.")

        watched_df = df[df["Titulo"].isin(watched_titles)]

        genres_watched = set()
        for g in watched_df["Genero"].dropna():
            for genre in g.split(","):
                genres_watched.add(genre.strip())

        preferred_type = watched_df["Tipo"].mode()[0]

        ##st.write("🎯 Géneros detectados en tu historial:")
        ##st.write(list(genres_watched))

        ##st.write(f"📌 Tipo de contenido preferido: **{preferred_type}**")

        if genres_watched:
            pattern = "|".join(genres_watched)
            recommendations = df[
                (~df["Titulo"].isin(watched_titles)) &
                (df["Tipo"] == preferred_type) &
                (df["Genero"].str.contains(pattern, case=False, na=False))
            ]
        else:
            recommendations = df[
                (~df["Titulo"].isin(watched_titles)) &
                (df["Tipo"] == preferred_type)
            ]

        recommendations = recommendations.sort_values(
            by="Clasificacion", ascending=False
        ).head(10)

        st.subheader("Recomendaciones para ti:")

        for _, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 3])

            with col1:
                st.image(row["link"], width=120)

            with col2:
                st.markdown(f"### {row['Titulo']}")
                st.write(f"**Tipo:** {row['Tipo']}")
                st.write(f"**Género:** {row['Genero']}")
                st.write(f"**Duración:** {row['Duracion']}")
                st.write(f"⭐ **Clasificación:** {row['Clasificacion']}")

            st.divider()

    elif username and not watched_titles:
        st.info("Selecciona al menos un título para generar recomendaciones.")

# ==================================================
# PAGE 3: TINDER STYLE (CORREGIDO)
# ==================================================
elif page == "Recomendación por match":

    st.markdown("<h1 style='text-align: center;'>🔥 Recomendación por match</h1>", unsafe_allow_html=True)

    # ---------- Inicializar estado ----------
    if "shown_titles" not in st.session_state:
        st.session_state.shown_titles = []
    if "liked_genres" not in st.session_state:
        st.session_state.liked_genres = set()
    if "current_item" not in st.session_state:
        st.session_state.current_item = None

    def get_new_item():
        available = df[~df["Titulo"].isin(st.session_state.shown_titles)]
        if available.empty: return None
        return available.sample(1).iloc[0]

    # Asignar item si no existe
    if st.session_state.current_item is None:
        st.session_state.current_item = get_new_item()

    item = st.session_state.current_item

    # ---------- CSS ----------
    st.markdown("""
        <style>
        .stButton button { width: 100%; height: 55px; font-size: 22px; border-radius: 12px; }
        div[data-testid="stImage"] { display: flex; justify-content: center; }
        </style>
    """, unsafe_allow_html=True)

    # ---------- Layout ----------
    c1, space1, c2, space2, c3 = st.columns([3, 0.6, 3, 0.6, 3])

    # =========================================================
    # COLUMNA 1: VOTACIÓN
    # =========================================================
    with c1:
        st.subheader("🎬 ¿Te gusta?")
        if item is None:
            st.success("🎉 No hay más contenido para mostrar")
            if st.button("Reiniciar"):
                st.session_state.shown_titles = []
                st.session_state.liked_genres = set()
                st.session_state.current_item = get_new_item()
                st.rerun()
        else:
            st.image(item["link"], use_container_width=True)
            st.markdown(f"## {item['Titulo']}")
            st.write(f"**Género:** {item['Genero']}")
            st.write(f"⭐ **Clasificación:** {item['Clasificacion']}")
            
            st.divider()
            col_x, col_v = st.columns(2)
            with col_x:
                if st.button("❌"):
                    st.session_state.shown_titles.append(item["Titulo"])
                    st.session_state.current_item = get_new_item()
                    st.rerun()
            with col_v:
                if st.button("💚"):
                    genres = [g.strip() for g in item["Genero"].split(",")]
                    st.session_state.liked_genres.update(genres)
                    st.session_state.shown_titles.append(item["Titulo"])
                    st.session_state.current_item = get_new_item()
                    st.rerun()

    # =========================================================
    # COLUMNA 2: RECOMENDACIONES
    # =========================================================
    with c2:
        st.subheader("🍿 Recomendaciones")
        if not st.session_state.liked_genres:
            st.info("👈 Dale a 💚 para ver sugerencias.")
        else:
            pattern = "|".join(st.session_state.liked_genres)
            
            # Validación de seguridad para item["Titulo"]
            current_title = item["Titulo"] if item is not None else ""
            
            recs_pop = df[
                (df["Genero"].str.contains(pattern, case=False, na=False)) &
                (df["Titulo"] != current_title)
            ].sort_values(by="Clasificacion", ascending=False).head(5)

            for _, r in recs_pop.iterrows():
                sub_c1, sub_c2 = st.columns([1, 3])
                with sub_c1: st.image(r["link"], width=100)
                with sub_c2:
                    st.markdown(f"### {r['Titulo']}")
                    st.write(f"**Género:** {r['Genero']}")
                    st.write(f"⭐ **Clasificación:** {r['Clasificacion']}")
                st.divider()

    # =========================================================
    # COLUMNA 3: JOYAS OCULTAS
    # =========================================================
    with c3:
        st.subheader("💎 Joyas Ocultas")
        if not st.session_state.liked_genres:
            st.info("Pelis poco conocidas.")
        else:
            pattern = "|".join(st.session_state.liked_genres)
            current_title = item["Titulo"] if item is not None else ""
            
            recs_hidden = df[
                (df["Genero"].str.contains(pattern, case=False, na=False)) &
                (df["Titulo"] != current_title) &
                (df["votes"] < 25)
            ].sort_values(by="Clasificacion", ascending=False).head(5)

            if recs_hidden.empty:
                st.write("No hay joyas ocultas disponibles.")
            else:
                for _, r in recs_hidden.iterrows():
                    sub_c1, sub_c2 = st.columns([1, 3])
                    with sub_c1: st.image(r["link"], width=100)
                    with sub_c2:
                        st.markdown(f"### {r['Titulo']}")
                        st.write(f"**Género:** {r['Genero']}")
                        st.write(f"⭐ **Clasificación:** {r['Clasificacion']}")
                    st.divider()


