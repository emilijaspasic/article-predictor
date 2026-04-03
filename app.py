import streamlit as st
import pickle
import Orange
import numpy as np

st.set_page_config(page_title="Article Performance Predictor", page_icon="📰", layout="centered")

st.title("📰 Article Performance Predictor")
st.markdown("Predict article performance and conversion likelihood before publication.")

@st.cache_resource
def load_models():
    with open("model1_performance.pkcls", "rb") as f:
        model1 = pickle.load(f)
    with open("model2_conversion.pkcls", "rb") as f:
        model2 = pickle.load(f)
    return model1, model2

model1, model2 = load_models()

st.subheader("Article details")

col1, col2 = st.columns(2)

with col1:
    publication = st.selectbox("Publication", ["Tages-Anzeiger", "24 heures", "BZ Berner Zeitung", "Basler Zeitung", "Der Bund", "Tribune de Genève"])
    category = st.selectbox("Category", ["sport", "politics", "economy", "lifestyle", "culture", "region", "entertainment", "knowledge", "magazine", "other", "service"])
    article_type = st.selectbox("Article type", ["agency", "newsletterarticle", "ticker", "slideshow"])
    user_needs = st.selectbox("User need", ["facts", "inspiration", "distract", "depth", "service"])
    sml_bin = st.selectbox("Format (SML)", ["XS", "S", "M", "L", "XL", "XXL", "Special Length", "3XL"])
    word_count = st.number_input("Word count", min_value=0, max_value=5000, value=500)
    expected_reading_time_mins = st.number_input("Expected reading time (mins)", min_value=0, max_value=70, value=3)
    day_of_week = st.selectbox("Day of week", ["1", "2", "3", "4", "5", "6", "7"])

with col2:
    published_at_month = st.selectbox("Month", ["1","2","3","4","5","6","7","8","9","10","11","12"])
    premium_flag = st.selectbox("Premium", ["1", "0"])
    discover_appearance_flag = st.selectbox("Google Discover", ["1", "0"])
    search_appearance_flag = st.selectbox("SEO / Search", ["1", "0"])
    editorial_origin = st.selectbox("Editorial origin", ["unknown", "reach", "mantel", "regional", "agency"])
    content_source = st.selectbox("Content source", ["Agency", "Editorial (Mantel + Regional)", "Reach", "Not Labelled", "Mantel", "Regional"])
    agency_flag = st.selectbox("Agency flag", ["1", "0"])
    is_ai_generated = st.selectbox("AI generated", ["True", "False"])

st.subheader("Content flags")
col3, col4, col5, col6 = st.columns(4)
with col3:
    is_opinion = st.selectbox("Opinion", ["True", "False"])
with col4:
    is_video_content = st.selectbox("Video", ["True", "False"])
with col5:
    is_podcast = st.selectbox("Podcast", ["True", "False"])
with col6:
    paywall_flag = st.selectbox("Paywall", ["yes", "no"])

st.subheader("Headline features")
col7, col8, col9 = st.columns(3)
with col7:
    title_word_count = st.number_input("Headline word count", min_value=0, max_value=30, value=8)
with col8:
    title_has_quote = st.selectbox("Has quote «»", ["yes", "no"])
with col9:
    title_has_number = st.selectbox("Has number", ["yes", "no"])

if st.button("Predict performance", type="primary"):

    try:
        domain1 = model1.domain
        domain2 = model2.domain

        vals1 = [
            category, word_count, article_type, user_needs,
            day_of_week, published_at_month, publication,
            is_podcast, is_video_content, agency_flag,
            content_source, sml_bin, is_opinion, is_ai_generated,
            editorial_origin, premium_flag, discover_appearance_flag,
            expected_reading_time_mins, search_appearance_flag,
            paywall_flag, title_word_count, title_has_quote, title_has_number
        ]

        vals2 = [
            article_type, agency_flag, word_count, sml_bin,
            user_needs, day_of_week, discover_appearance_flag,
            search_appearance_flag, expected_reading_time_mins,
            editorial_origin, is_ai_generated, is_opinion,
            is_video_content, is_podcast, content_source,
            category, publication
        ]

        instance1 = Orange.data.Instance(domain1, vals1)
        instance2 = Orange.data.Instance(domain2, vals2)

        pred1 = model1(instance1)
        pred2 = model2(instance2)

        perf = domain1.class_var.values[int(pred1)]
        conv = domain2.class_var.values[int(pred2)]

        st.divider()
        st.subheader("Prediction results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weekly performance", perf)
        with col2:
            st.metric("Conversion", conv)

        if perf == "Top 25%" and conv == "converted":
            st.success("🏆 Ideal article — high performance and high conversion potential!")
        elif perf == "Top 25%" and conv == "not_converted":
            st.info("👁 Reach article — high traffic but low conversion. Add premium hooks.")
        elif perf != "Top 25%" and conv == "converted":
            st.warning("⭐ Niche converter — loyal audience, high intent. Protect this content.")
        else:
            st.error("⚠️ High risk — reconsider format, length or user need.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Make sure all fields match the values used during model training.")
