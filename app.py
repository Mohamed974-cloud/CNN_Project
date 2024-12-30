import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Maladies des Pommes de Terre",
    layout="wide"
)

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modele_pommes_de_terre.h5')

# Fonction de prétraitement des images
def preprocess_image(img):
    # Convertir explicitement l'image en RGB
    # La profondeur d'entrée doit être un multiple de la profondeur du filtre : 4 vs 3"
    img = img.convert('RGB')
    
    # Redimensionner
    img = img.resize((224, 224))
    
    # Convertir en array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Normaliser
    img_array = img_array / 255.0
    
    # Vérifier la forme de l'array
    if img_array.shape[-1] != 3:
        raise ValueError("L'image doit avoir 3 canaux (RGB)")
    
    # Ajouter la dimension du batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
    # # Redimensionner l'image à la taille attendue par le modèle
    # img = img.resize((224, 224))
    # # Convertir en array et normaliser
    # img_array = tf.keras.preprocessing.image.img_to_array(img)
    # img_array = img_array / 255.0
    # img_array = np.expand_dims(img_array, axis=0)
    # return img_array

# Interface principale
def main():
    st.title("Détecteur de Maladies des Pommes de Terre 🥔")
    
    st.write("""
    Cette application utilise l'intelligence artificielle pour détecter si une pomme 
    de terre est saine ou malade à partir d'une image.
    """)
    
    # Sidebar avec les informations
    st.sidebar.title("À propos")
    st.sidebar.info(
        "Cette application utilise un modèle CNN entraîné sur des images "
        "de pommes de terre pour détecter les maladies."
    ) 
    
    # Zone de téléchargement d'image
    uploaded_file = st.file_uploader(
        "Choisissez une image de pomme de terre...",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        try:
            # Afficher l'image originale
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Image téléchargée")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
            # Prétraitement et prédiction
            processed_image = preprocess_image(image)
            model = load_model()
            prediction = model.predict(processed_image)
            
            with col2:
                st.subheader("Résultats de l'analyse")
                
                # Afficher les probabilités
                # prob_saine = prediction[0][0] * 100
                # prob_malade = prediction[0][1] * 100
                prob_saine = float(prediction[0][0] * 100)
                prob_malade = float(prediction[0][1] * 100)
                
                st.write("Probabilités :")
                st.progress(prob_saine / 100)
                st.write(f"Saine : {prob_saine:.2f}%")
                
                st.progress(prob_malade / 100)
                st.write(f"Malade : {prob_malade:.2f}%")
                
                # Afficher la conclusion
                st.write("---")
                if prob_saine > prob_malade:
                    st.success("✅ La pomme de terre semble être saine!")
                else:
                    st.error("❌ La pomme de terre semble être malade!")
                
        except Exception as e:
            st.error(f"Une erreur s'est produite : {str(e)}")

if __name__ == "__main__":
    main()