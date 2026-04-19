import streamlit as st
import pandas as pd
import joblib
import json
import os

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Gurgaon Real Estate Analytics", page_icon="🏡", layout="wide")
st.title("🏡 Intelligent Property Price Predictor (Gurgaon)")
st.markdown("Enter the property specifications below to generate an AI-driven valuation and advisory report.")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.warning("⚠️ Please add GROQ_API_KEY to your Streamlit secrets!")

@st.cache_resource(show_spinner="Loading ML Model and Vector Database...")
def load_resources():
    ml_pipeline = joblib.load('rf_gurgaon_pipeline_final.pkl')
    
    df = pd.read_csv('gurgaon_properties_post_feature_selection_v2.csv')
    df_rag = df.copy()
    df_rag['rag_content'] = df_rag.apply(
        lambda row: f"{row['bedRoom']} BHK {row['property_type']} in {row['sector']}. Price: {row['price']} Crores. Area: {row['built_up_area']} sqft. Age: {row['agePossession']}. Luxury Category: {row['luxury_category']}.", 
        axis=1
    )
    loader = DataFrameLoader(df_rag, page_content_column="rag_content")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(loader.load(), embeddings)
    
    return ml_pipeline, vector_store

pipeline, vector_store = load_resources()

st.header("Property Characteristics")
col1, col2, col3 = st.columns(3)

with col1:
    property_type = st.selectbox("Property Type", ['flat', 'house'])
    sector = st.text_input("Sector (e.g., 'sector 36')", value='sector 36').lower()
    built_up_area = st.number_input("Built-up Area (Sq. Ft.)", min_value=100.0, value=1200.0, step=50.0)
    agePossession = st.selectbox("Age of Property", ['New Property', 'Relatively New', 'Moderately Old', 'Old Property', 'Under Construction'])

with col2:
    bedRoom = st.number_input("Bedrooms", min_value=1.0, value=3.0, step=1.0)
    bathroom = st.number_input("Bathrooms", min_value=1.0, value=2.0, step=1.0)
    balcony = st.selectbox("Balconies", ['0', '1', '2', '3', '3+'])
    floor_category = st.selectbox("Floor Category", ['Low Floor', 'Mid Floor', 'High Floor'])

with col3:
    luxury_category = st.selectbox("Luxury Category", ['Low', 'Medium', 'High'])
    furnishing_type = st.selectbox("Furnishing (0=Unf, 1=Semi, 2=Full)", [0.0, 1.0, 2.0])
    servant_room = st.selectbox("Servant Room", [0.0, 1.0])
    store_room = st.selectbox("Store Room", [0.0, 1.0])

input_data = pd.DataFrame([[
    property_type, sector, bedRoom, bathroom, balcony, agePossession, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category
]], columns=[
    'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony', 'agePossession', 'built_up_area', 'servant room', 'store room', 'furnishing_type', 'luxury_category', 'floor_category'
])

btn_col1, btn_col2 = st.columns([1, 1])

with btn_col1:
    if st.button("Generate Valuation", type="primary", use_container_width=True):
        try:
            prediction = pipeline.predict(input_data)[0]
            st.success("Analytics Engine execution complete.")
            st.metric(label="Estimated Market Value", value=f"₹ {prediction:.2f} Crores")
        except Exception as e:
            st.error(f"Error processing prediction: {e}")

with btn_col2:
    if st.button("Generate AI Advisory Report", type="secondary", use_container_width=True):
        if "GROQ_API_KEY" not in os.environ:
            st.error("Missing Groq API Key. Please add it to Streamlit Secrets.")
        else:
            with st.spinner("Calculating price, fetching comps, and writing report..."):
                try:
                    prediction = pipeline.predict(input_data)[0]
                    
                    search_query = f"{bedRoom} BHK {property_type} in {sector} luxury {luxury_category}"
                    retrieved_docs = vector_store.similarity_search(search_query, k=3)
                    comps_text = "\n".join([doc.page_content for doc in retrieved_docs])
                    
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile",
                        temperature=0.2, 
                        model_kwargs={"response_format": {"type": "json_object"}} 
                    )
                    
                    system_prompt = """
                    You are an elite real estate advisor in Gurgaon.
                    Always respond ONLY in this strictly formatted JSON object:
                    {
                      "summary": "Executive overview of the property and valuation.",
                      "comps": "Analysis of how this property compares to the real comparables.",
                      "recommendation": "Buy/Sell/Hold advice with justification.",
                      "risk": "Potential market or property risks.",
                      "disclaimer": "Standard real estate disclaimer."
                    }
                    """
                    
                    human_prompt = f"""Target Property: {bedRoom} BHK {property_type}, {sector}, {built_up_area} sqft, {agePossession}, {luxury_category} luxury. AI Value: ₹ {prediction:.2f} Crores. Comparables: {comps_text}"""
                    
                    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
                    report_data = json.loads(response.content)
                    
                    st.success(f"Report Generated! Estimated Value: ₹ {prediction:.2f} Crores")
                    
                    with st.expander("📄 Executive Summary", expanded=True): st.write(report_data.get("summary", ""))
                    with st.expander("📊 Market Comparables (Comps)"):
                        st.write(report_data.get("comps", ""))
                        st.info(f"**Raw DB Data:**\n{comps_text}")
                        
                    colA, colB = st.columns(2)
                    with colA:
                        st.subheader("💡 Recommendation")
                        st.info(report_data.get("recommendation", ""))
                    with colB:
                        st.subheader("⚠️ Risk Assessment")
                        st.warning(report_data.get("risk", ""))
                        
                    st.caption(f"**Disclaimer:** {report_data.get('disclaimer', '')}")
                    
                except Exception as e:
                    st.error(f"Error generating report: {e}")
