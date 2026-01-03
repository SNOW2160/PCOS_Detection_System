import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# PASTE YOUR GEMINI API KEY HERE
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("API Key not found! Please set it in .streamlit/secrets.toml")
    st.stop()

st.set_page_config(page_title="PCOS Detection System", page_icon="🔬", layout="wide")

@st.cache_resource
def load_lifestyle_model():
    try:
        model = joblib.load("models/xgb_pcos_lifestyle.pkl")
        feat_cols = joblib.load("models/feature_names.pkl")
        return model, feat_cols
    except Exception as e:
        st.error(f"Error loading XGBoost: {e}")
        return None, None

@st.cache_resource
def load_image_model():
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load("models/resnet18_pcos.pth", 
                                        map_location=torch.device('cpu'), 
                                        weights_only=True))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading ResNet: {e}")
        return None

@st.cache_resource
def load_rag_chain():
    if "PASTE" in GOOGLE_API_KEY:
        st.warning("Google API Key not set in app.py. RAG will not work.")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("models/faiss_pcos_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
        
        template = """You are an expert PCOS Consultant. Use the following Medical Guidelines to answer.

CONTEXT: {context}

PATIENT DATA: {question}

INSTRUCTIONS:
1. Analyze the patient's data against the guidelines.
2. Provide 3 specific, actionable steps based on the text.
3. If the answer is not in the context, say "Please consult a specialist."

YOUR DIAGNOSIS PLAN:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {"context": vectorstore.as_retriever(search_kwargs={"k": 3}) | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
    except Exception as e:
        st.error(f"Error loading RAG: {e}")
        return None

# Initialize Models
xgb_model, feature_cols = load_lifestyle_model()
cnn_model = load_image_model()
rag_chain = load_rag_chain()

def preprocess_image(image):
    # FIXED: Explicit tuple size=(224, 224), interpolation=InterpolationMode.BILINEAR for new torchvision
    from torchvision.transforms.functional import InterpolationMode
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- UI LAYOUT ---
st.title("🔬 Holistic PCOS Diagnosis")
st.markdown("Combines Ultrasound Imaging + Lifestyle Data + Medical Guidelines (RAG)")

st.sidebar.header("Patient Profile")
age = st.sidebar.number_input("Age", 10, 60, 25)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 24.0)
cycle_len = st.sidebar.number_input("Cycle Length (Days)", 20, 90, 28)
cycle_reg = st.sidebar.selectbox("Cycle Regularity", ["Regular", "Irregular"])
weight_gain = st.sidebar.checkbox("Weight Gain?")
hirsutism = st.sidebar.checkbox("Excess Hair Growth?")
skin_dark = st.sidebar.checkbox("Skin Darkening?")
pimples = st.sidebar.checkbox("Acne/Pimples?")
fast_food = st.sidebar.checkbox("Frequent Fast Food?")
exercise = st.sidebar.checkbox("Regular Exercise?")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Ultrasound", type=["jpg", "png", "jpeg"])

if st.button("Analyze Health Risk", type="primary"):
    # A. Lifestyle Prediction
    input_data = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Cycle_Length": [cycle_len],
        "Cycle_Regularity": [1 if cycle_reg == "Irregular" else 0],
        "Weight_Gain": [int(weight_gain)],
        "Hirsutism": [int(hirsutism)],
        "Skin_Darkening": [int(skin_dark)],
        "Pimples": [int(pimples)],
        "Fast_Food": [int(fast_food)],
        "Exercise": [int(exercise)]
    })
    
    # Safely select columns
    if feature_cols is not None and xgb_model is not None:
        available_cols = [col for col in feature_cols if col in input_data.columns]
        if len(available_cols) == len(feature_cols):
            input_data = input_data[feature_cols]
            risk_lifestyle = xgb_model.predict_proba(input_data)[:, 1][0]
        else:
            st.error(f"Missing columns: {set(feature_cols) - set(input_data.columns)}. Check feature_names.pkl.")
            st.stop()
    else:
        st.error("XGBoost model or features not loaded.")
        st.stop()

    # B. Image Prediction
    risk_image = None
    if uploaded_file and cnn_model is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            img_t = preprocess_image(img)
            with torch.no_grad():
                out = cnn_model(img_t)
                prob = torch.nn.functional.softmax(out, dim=1)
            risk_image = prob[:, 0].item()  # Assuming Class 0 = PCOS
        except Exception as e:
            st.error(f"Image processing error: {e}")

    # C. Fusion
    if risk_image is not None:
        final_risk = 0.4 * risk_lifestyle + 0.6 * risk_image
        method = "Multi-Modal Fusion (Img + Data)"
    else:
        final_risk = risk_lifestyle
        method = "Lifestyle Only"

    risk_percent = final_risk * 100
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Results")
        st.info(f"Method Used: {method}")
        if final_risk > 0.5:
            st.markdown(f"<h2 style='color:red'>High Risk: {risk_percent:.1f}%</h2>", unsafe_allow_html=True)
            st.error("Status: Potential PCOS Detected")
        else:
            st.markdown(f"<h2 style='color:green'>Low Risk: {risk_percent:.1f}%</h2>", unsafe_allow_html=True)
            st.success("Status: Healthy Range")
    
    with col2:
        if uploaded_file:
            st.image(uploaded_file, caption="Scan Analyzed", width=250)

    # E. RAG Recommendations
    if final_risk > 0.5:
        st.divider()
        st.subheader("AI Clinical Recommendations")
        with st.spinner("System is consulting the medical guidelines..."):
            patient_desc = f"""Patient Details:
- BMI: {bmi}
- Irregular Cycles: {cycle_reg}
- Hirsutism: {'Yes' if hirsutism else 'No'}
- Acne: {'Yes' if pimples else 'No'}
- Exercise: {'Yes' if exercise else 'No'}
- Fast Food: {'High' if fast_food else 'Low'}
Question: What are the most important lifestyle changes for this patient?"""
            if rag_chain:
                response = rag_chain.invoke(patient_desc)
                st.write(response)
            else:
                st.warning("RAG System not active. Check API Key.")
