import os
import streamlit as st
import pandas as pd
from PIL import Image
import random
import streamlit.components.v1 as components
from combinedModels import HybridHotelRecommender
import requests

BASE_PATH = "C:/Users/HP/Desktop/IAII_M2/SystèmeDeRecommendation/projet/combinedModel/"
DATA_PATH = "C:/Users/HP/Desktop/IAII_M2/SystèmeDeRecommendation/projet/RSData/maData/"


UNSPLASH_ACCESS_KEY = "aQOxJl_2HV0rFDhnFCWjtkvpxRny9mfMgRuv4-yId8w"

def fetch_image_url(query):
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "client_id": UNSPLASH_ACCESS_KEY,
        "per_page": 1
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["results"]:
                return data["results"][0]["urls"]["regular"]
    except Exception as e:
        print(f"Erreur lors de la récupération de l'image : {e}")
    return "https://via.placeholder.com/400"  # Image par défaut

def load_data():
    offerings_df = pd.read_csv(os.path.join(DATA_PATH, 'offering.csv'))
    user_data = pd.read_csv(os.path.join(DATA_PATH, 'reviews.csv'))
    
    user_data['author_id'] = user_data['author_id'].astype(str)
    
    offerings_df = offerings_df.rename(columns={
        'name': 'hotel_name',
        'locality': 'city',
        'hotel_class': 'price',
    })
    
    hotel_ratings = user_data.groupby('offering_id')['overall_rating'].mean().reset_index()
     
    # Merger avec la bonne colonne
    if 'id' in offerings_df.columns:
        offerings_df = offerings_df.merge(hotel_ratings, left_on='id', right_on='offering_id', how='left')
    else:
        offerings_df['id'] = offerings_df.index
        offerings_df = offerings_df.merge(hotel_ratings, left_on='id', right_on='offering_id', how='left')
    
    offerings_df = offerings_df.rename(columns={'overall_rating': 'rating'})
    offerings_df['rating'] = offerings_df['rating'].fillna(0)
    
    return offerings_df, user_data
    return offerings_df, user_data

def login_page():
    st.session_state.page = "login"
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <style>
                .login-box {
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    background-color: white;
                    margin: 2rem 0;
                }
            </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.title("🏨 Hotel Recommender")
            user_id = st.text_input("Entrez votre ID utilisateur")
            if st.button("Connexion", use_container_width=True):
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.page = "main"
                    st.rerun()
                else:
                    st.error("Veuillez entrer un ID utilisateur")

def hotel_card(hotel):
    image_url = fetch_image_url(hotel['hotel_name'])  
    card_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; margin: 10px 0; background: #262730; width: 100%; height: 400px; overflow: hidden;">
        <div style="height: 200px; background: linear-gradient(45deg, #f3f4f6, #e5e7eb); overflow: hidden;">
            <img src="{image_url}" alt="{hotel['hotel_name']}" style="width: 100%; height: 100%; object-fit: cover;">
        </div>
        <div style="padding: 15px;">
            <h3 style="font-size: 1.2rem; margin: 0 0 10px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{hotel['hotel_name']}</h3>
            <p style="margin: 8px 0;">📍 {hotel['city']}</p>
            <p style="margin: 8px 0;">⭐ Classe: {hotel['price']}</p>
            <p style="margin: 8px 0;">📊 Note: {hotel['rating']:.1f}/5</p>
        </div>
    </div>
    """
    return card_html


def main_page(offerings_df, user_data, recommender):
    st.title(f"Bienvenue sur Hotel Recommender")
    
    tab1, tab2 = st.tabs(["Catalogue", "Recommandations"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            price_range = st.slider("Classe d'hôtel", 1.0, 5.0, (1.0, 5.0))
        with col2:
            rating_filter = st.slider("Note minimum", 0.0, 5.0, 0.0, 0.5)
        with col3:
            cities = ['Toutes les villes'] + sorted(offerings_df['city'].unique().tolist())
            selected_city = st.selectbox("Ville", cities)

        # Filtrer les hôtels selon les critères
        filtered_df = offerings_df[(
            offerings_df['price'] >= price_range[0]) & 
            (offerings_df['price'] <= price_range[1]) & 
            (offerings_df['rating'] >= rating_filter)
        ]
        if selected_city != 'Toutes les villes':
            filtered_df = filtered_df[filtered_df['city'] == selected_city]
        
        # Pagination
        hotels_per_page = 10
        total_hotels = len(filtered_df)
        total_pages = (total_hotels - 1) // hotels_per_page + 1
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1

        # Afficher les hôtels de la page actuelle
        cols = st.columns(3)
        for idx, (_, hotel) in enumerate(filtered_df.iloc[
            (st.session_state.current_page - 1) * hotels_per_page : st.session_state.current_page * hotels_per_page
        ].iterrows()):
            with cols[idx % 3]:
                st.markdown(hotel_card(hotel), unsafe_allow_html=True)

        # Indicateur de page
        st.markdown(f"Page {st.session_state.current_page} sur {total_pages}")

        # Flèches de pagination sous l'indicateur
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️", key="prev_page") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
        with col3:
            if st.button("➡️", key="next_page") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1

    with tab2:
        if st.button("Obtenir les recommandations"):
            with st.spinner('Génération des recommandations...'):
                try:
                    offerings2_df = pd.read_csv(os.path.join(DATA_PATH, 'offering2.csv'))
                    user_data_adapted = user_data.copy()
                    user_data_adapted['user_id'] = user_data_adapted['author_id']
                    
                    recommendations = recommender.get_hybrid_recommendations(
                        user_id=st.session_state.user_id,
                        user_data=user_data_adapted,
                        offerings_df=offerings2_df,
                        n_recommendations=5
                    )
                    
                    if recommendations is not None and not recommendations.empty:
                        recommended_hotels = recommendations['hotel_id'].tolist()
                        display_df = offerings_df[offerings_df['id'].isin(recommended_hotels)]
                        
                        cols = st.columns(3)
                        for idx, (_, hotel) in enumerate(display_df.iterrows()):
                            with cols[idx % 3]:
                                st.markdown(hotel_card(hotel), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")

def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = "login"
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

def main():
    st.set_page_config(layout="wide", page_title="Hotel Recommender")
    
    init_session_state()
    
    offerings_df, user_data = load_data()
    recommender = HybridHotelRecommender.load_model('hybrid_recommender_model.pkl')
    
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; padding: 20px;">
""", unsafe_allow_html=True)
    
    if st.session_state.page == "login":
        login_page()
    else:
        main_page(offerings_df, user_data, recommender)

if __name__ == "__main__":
    main()