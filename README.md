PCOS Detection System

This project is a multi-modal AI-based system designed for the early detection and risk assessment of Polycystic Ovary Syndrome (PCOS). It combines lifestyle and clinical data, ultrasound image analysis, and medical guideline–based recommendations to provide a holistic and explainable diagnosis.

The system uses a machine learning model (XGBoost) to analyze lifestyle and clinical parameters such as BMI, menstrual cycle regularity, and symptoms. A deep learning model (ResNet-18 CNN) is used to analyze ultrasound images. The outputs from both models are combined using a late fusion strategy to generate a final PCOS risk score.

In addition to risk prediction, the system integrates a Retrieval-Augmented Generation (RAG) module that uses medical guidelines stored in a vector database (FAISS) and a large language model (Google Gemini) to provide personalized, clinically grounded lifestyle and health recommendations.

Technologies Used:
- Python
- Streamlit
- PyTorch
- XGBoost
- LangChain
- FAISS
- Google Gemini
- Pandas, NumPy

Main Features:
- Lifestyle-based PCOS risk prediction using XGBoost
- Ultrasound image classification using CNN (ResNet-18)
- Multi-modal late fusion for improved accuracy
- AI-generated medical recommendations using RAG
- Interactive web interface built with Streamlit

How to Run:
1. Install dependencies using: pip install -r requirements.txt
2. Set your Google API key in .streamlit/secrets.toml
3. Run the application using: streamlit run app.py

Note:
This project is intended for educational and research purposes only and does not replace professional medical diagnosis.

Author:
Manav Patode
