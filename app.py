import streamlit as st
import pandas as pd
import requests
import json
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
from google_play_scraper import reviews, Sort
from google.api_core.exceptions import ResourceExhausted
from wordcloud import WordCloud
import numpy as np

# --- LOAD SECRETS ---
# Try Streamlit secrets first (for deployed app), fallback to environment variables
try:
    ENV_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    ENV_SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback to environment variables for local development
    ENV_GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    ENV_SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# --- PAGE CONFIG ---
st.set_page_config(page_title="App Sentiment Analyzer", layout="wide")
st.title("📱 Aspect-Based App Sentiment Analyzer")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("Configuration")
    
    # Add toggle for custom keys
    use_custom_keys = st.checkbox("Use my own API keys", value=False)
    
    if use_custom_keys:
        gemini_key = st.text_input("Gemini API Key", type="password", key="custom_gemini")
        serpapi_key = st.text_input("SerpApi Key", type="password", key="custom_serpapi")
        
        if gemini_key and serpapi_key:
            st.success("✅ Using your API keys")
        elif not gemini_key:
            st.warning("⚠️ Please enter your Gemini API key")
        elif not serpapi_key:
            st.warning("⚠️ Please enter your SerpApi key")
    else:
        # Use default keys from environment
        if ENV_GEMINI_KEY and ENV_SERPAPI_KEY:
            gemini_key = ENV_GEMINI_KEY
            serpapi_key = ENV_SERPAPI_KEY
            st.success("✅ Using default API keys")
        else:
            st.error("❌ Default API keys not configured")
            gemini_key = None
            serpapi_key = None
    
    if gemini_key:
        genai.configure(api_key=gemini_key)
    
    st.divider()
    st.info("Using Gemini 2.5 Flash for analysis.")

# --- FUNCTIONS ---

def search_app_serpapi(query, api_key):
    params = {
        "engine": "google_play",
        "q": query,
        "api_key": api_key,
        "store": "apps"
    }
    try:
        response = requests.get("https://serpapi.com/search.json", params=params)
        data = response.json()
        
        # Debug: show what we got back
        if "error" in data:
            st.error(f"SerpApi returned error: {data['error']}")
            return None
        
        # Priority 1: Check app_highlight (the top matched result)
        if "app_highlight" in data:
            top_result = data["app_highlight"]
        else:
            # Priority 2: Check items array
            results = data.get("items", [])
            if not results:
                # Priority 3: Check organic_results with items nested
                organic = data.get("organic_results", [])
                if organic and len(organic) > 0 and "items" in organic[0]:
                    results = organic[0]["items"]
            
            if not results or len(results) == 0:
                st.warning(f"No results found for '{query}'. Try a different search term.")
                with st.expander("🔍 Debug: Full SerpApi Response"):
                    st.json(data)
                return None
            
            top_result = results[0]
        
        # Extract fields
        app_id = top_result.get("product_id")
        title = top_result.get("title")
        thumbnail = top_result.get("thumbnail")
        
        # Debug output
        with st.expander("🔍 Debug: Selected App"):
            st.json(top_result)
        
        if not app_id or not title:
            st.warning("App found but missing required fields. Try a more specific search.")
            with st.expander("🔍 Debug: Full Response"):
                st.json(data)
            return None
            
        return {
            "title": title,
            "app_id": app_id,
            "thumbnail": thumbnail
        }
    except Exception as e:
        st.error(f"SerpApi Error: {e}")
        return None

def scrape_play_store(app_id, count=100):
    result, _ = reviews(
        app_id,
        lang='en',
        country='us',
        sort=Sort.NEWEST,
        count=count
    )
    df = pd.DataFrame(result)
    if not df.empty:
        df = df.rename(columns={'reviewId': 'review_id', 'content': 'review', 'score': 'rating'})
    return df

def safe_gemini_call(model, prompt):
    retries = 3
    for _ in range(retries):
        try:
            response = model.generate_content(prompt)
            return json.loads(response.text)
        except ResourceExhausted:
            time.sleep(10)
        except Exception:
            time.sleep(2)
    return {}

# --- MAIN UI ---

col1, col2 = st.columns([3, 1])
with col1:
    app_name = st.text_input("Enter App Name (e.g., Spotify)")
with col2:
    review_count = st.selectbox("Reviews to Scrape", [10, 20, 50, 100, 250, 500, 1000])

if st.button("Find & Analyze App"):
    if not gemini_key or not serpapi_key:
        st.error("Please enter both API keys in the sidebar.")
    else:
        with st.status("Searching Google Play...", expanded=True) as status:
            app_data = search_app_serpapi(app_name, serpapi_key)
            
            if app_data:
                st.write(f"✅ Found App: **{app_data['title']}**")
                if app_data['thumbnail']:
                    st.image(app_data['thumbnail'], width=100)
                else:
                    st.info("No thumbnail available")
                st.write(f"App ID: `{app_data['app_id']}`")
                
                # Only proceed if we have a valid app_id
                if not app_data['app_id']:
                    st.error("Cannot proceed without a valid app ID.")
                    st.stop()
                
                st.write(f"Scraping {review_count} reviews...")
                df = scrape_play_store(app_data['app_id'], count=review_count)
                st.write(f"✅ Scraped {len(df)} reviews.")
                
                st.write("Running Gemini Analysis Pipeline...")
                progress_bar = st.progress(0)
                
                model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
                results = []
                
                for i, row in df.iterrows():
                    progress_bar.progress((i + 1) / len(df))
                    
                    # --- FIX 1: Add Sleep BEFORE Extraction to prevent rate limits ---
                    time.sleep(2) 
                    
                    text = row['review']
                    aspect_prompt = f"Extract technical aspects (e.g. UI, Audio) from this review: '{text}'. Return JSON: {{'aspects': ['...']}}"
                    aspect_data = safe_gemini_call(model, aspect_prompt)
                    aspects = aspect_data.get('aspects', [])
                    
                    if aspects:
                        for aspect in aspects:
                            time.sleep(4) # Slightly reduced since we added the pre-sleep
                            sent_prompt = f"Sentiment of '{aspect}' in '{text}'? Return JSON: {{'sentiment': 'Positive/Negative/Neutral'}}"
                            sent_data = safe_gemini_call(model, sent_prompt)
                            
                            results.append({
                                'aspect': aspect,
                                'sentiment': sent_data.get('sentiment', 'Neutral')
                            })
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                if results:
                    results_df = pd.read_json(json.dumps(results))
                    results_df['aspect'] = results_df['aspect'].str.title()
                    results_df['sentiment'] = results_df['sentiment'].str.title()
                    
                    st.divider()
                    
                    # --- Word Cloud ---
                    st.subheader("📊 Review Word Cloud")
                    review_text = ' '.join(df['review'].dropna().astype(str))
                    if review_text.strip():
                        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                            colormap='viridis', max_words=100).generate(review_text)
                        fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
                        ax_wc.imshow(wordcloud, interpolation='bilinear')
                        ax_wc.axis('off')
                        st.pyplot(fig_wc)
                    else:
                        st.info("Not enough text data to generate word cloud.")
                    
                    st.divider()
                    
                    st.subheader("1. Most Discussed Aspects")
                    top_aspects = results_df['aspect'].value_counts().head(10)
                    fig1, ax1 = plt.subplots(figsize=(10, 5))
                    sns.barplot(x=top_aspects.values, y=top_aspects.index, ax=ax1, palette='viridis')
                    ax1.set_xlabel('Mention Count')
                    ax1.set_ylabel('Aspect')
                    st.pyplot(fig1)
                    
                    # --- NEW: Most Loved Aspects ---
                    st.subheader("2. Most Loved Aspects ❤️")
                    positive_df = results_df[results_df['sentiment'] == 'Positive']
                    if not positive_df.empty:
                        loved_aspects = positive_df['aspect'].value_counts().head(10)
                        fig_loved, ax_loved = plt.subplots(figsize=(10, 5))
                        sns.barplot(x=loved_aspects.values, y=loved_aspects.index, ax=ax_loved, palette='Greens_r')
                        ax_loved.set_xlabel('Positive Mentions')
                        ax_loved.set_ylabel('Aspect')
                        st.pyplot(fig_loved)
                    else:
                        st.info("No positive aspects found in the reviews.")
                    
                    # --- NEW: Most Criticized Aspects ---
                    st.subheader("3. Most Criticized Aspects 👎")
                    negative_df = results_df[results_df['sentiment'] == 'Negative']
                    if not negative_df.empty:
                        criticized_aspects = negative_df['aspect'].value_counts().head(10)
                        fig_criticized, ax_criticized = plt.subplots(figsize=(10, 5))
                        sns.barplot(x=criticized_aspects.values, y=criticized_aspects.index, ax=ax_criticized, palette='Reds_r')
                        ax_criticized.set_xlabel('Negative Mentions')
                        ax_criticized.set_ylabel('Aspect')
                        st.pyplot(fig_criticized)
                    else:
                        st.info("No negative aspects found in the reviews.")
                    
                    # --- Aspect-Sentiment Heatmap ---
                    st.subheader("4. Aspect-Sentiment Heatmap 🔥")
                    heatmap_data = pd.crosstab(results_df['aspect'], results_df['sentiment'])
                    
                    # Only show top 15 aspects for readability
                    top_15_aspects = results_df['aspect'].value_counts().head(15).index
                    heatmap_data = heatmap_data.loc[top_15_aspects]
                    
                    if not heatmap_data.empty:
                        fig_heat, ax_heat = plt.subplots(figsize=(10, 8))
                        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
                                   cbar_kws={'label': 'Count'}, ax=ax_heat)
                        ax_heat.set_xlabel('Sentiment')
                        ax_heat.set_ylabel('Aspect')
                        plt.xticks(rotation=0)
                        plt.yticks(rotation=0)
                        st.pyplot(fig_heat)
                    else:
                        st.info("Not enough data to generate heatmap.")
                    
                    # --- FIX 2: Stable Colors for Stacked Chart ---
                    st.subheader("5. Sentiment Distribution")
                    top_names = top_aspects.index.tolist()
                    filtered = results_df[results_df['aspect'].isin(top_names)]
                    ct = pd.crosstab(filtered['aspect'], filtered['sentiment'], normalize='index') * 100
                    
                    # Ensure columns exist in specific order so colors match
                    desired_order = ['Negative', 'Neutral', 'Positive']
                    existing_cols = [c for c in desired_order if c in ct.columns]
                    ct = ct[existing_cols]
                    
                    # Map colors dynamically based on what columns actually exist
                    color_map = {'Negative': '#ff4d4d', 'Neutral': '#d3d3d3', 'Positive': '#4da6ff'}
                    final_colors = [color_map[c] for c in existing_cols]
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ct.plot(kind='bar', stacked=True, color=final_colors, ax=ax2)
                    ax2.set_xlabel('Aspect')
                    ax2.set_ylabel('Percentage (%)')
                    ax2.legend(title='Sentiment')
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig2)
                    
                    with st.expander("View Raw Data"):
                        st.dataframe(results_df)
                else:
                    st.warning("No aspects found in these reviews.")
            else:
                st.error("App not found. Please try a different name.")
                st.stop()