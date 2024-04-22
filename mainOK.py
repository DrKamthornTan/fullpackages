import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import os
from tqdm import tqdm

# First, we load the respective CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Directory containing the images
img_folder = 'C:\\Users\\kamth\\packages2'

use_precomputed_embeddings = True
emb_filename = 'photos-embeddings.pkl'

if use_precomputed_embeddings and os.path.exists(emb_filename):
    with open(emb_filename, 'rb') as fIn:
        img_names, img_emb = pickle.load(fIn)
    st.write("Images:", len(img_names))
else:
    img_names = list(glob.glob(os.path.join(img_folder, '*.png')))
    st.write("Images:", len(img_names))
    img_emb = torch.stack([model.encode(Image.open(filepath), convert_to_tensor=True) for filepath in tqdm(img_names)])

    if use_precomputed_embeddings:
        with open(emb_filename, 'wb') as fOut:
            pickle.dump((img_names, img_emb), fOut)

# Streamlit app
def main():
    st.title("DHV AI Startup Packages Search Demo")

    # Text input
    query = st.text_input("โปรดใส่ข้อมูลแพ็คเกจ")

    # Image search button
    if st.button("ค้นหาแพ็คเกจ"):
        if query == "":
            st.error("โปรดใส่ข้อมูลที่ต้องการค้นหา")
        else:
            search(query)

def search(query, k=3):
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)[0]  # Extract the tensor from the list

    # Reshape query embedding to match the expected input shape
    query_emb = query_emb.unsqueeze(0)  # Add a batch dimension

    # Reshape image embeddings to match the expected input shape
    img_emb_reshaped = img_emb.view(img_emb.shape[0], -1)

    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, img_emb_reshaped, top_k=k)[0]  

    st.text("ผลการค้นหา:")
    st.text(query)
    for hit in hits:
        #st.text(hit['corpus_id'])
        st.image(img_names[hit['corpus_id']], width=200)


if __name__ == "__main__":
    main()

