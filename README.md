# 📱 Aspect-Based App Sentiment Analyzer

A powerful Streamlit application that performs aspect-based sentiment analysis on Google Play Store app reviews using Google's Gemini 2.5 Flash AI model.

## 🌟 Features

- **Smart App Search**: Find any app on Google Play Store by name
- **Automated Review Scraping**: Collect 10-1000 reviews from the Play Store
- **AI-Powered Analysis**: Extract technical aspects (UI, Performance, Features, etc.) and their sentiments
- **Rich Visualizations**:
  - Word Cloud of reviews
  - Most discussed aspects
  - Most loved aspects ❤️
  - Most criticized aspects 👎
  - Aspect-sentiment heatmap
  - Sentiment distribution charts
- **Flexible API Key Management**: Use default keys or provide your own

## 🚀 Demo

Search for any app (e.g., "Spotify", "Instagram", "WhatsApp") and get instant insights into what users love and hate about it!

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API Key ([Get it here](https://makersuite.google.com/app/apikey))
- SerpApi Key ([Get it here](https://serpapi.com/))

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up secrets for local development**
   
   Create a `.streamlit/secrets.toml` file:
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key-here"
   SERPAPI_KEY = "your-serpapi-key-here"
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## 🔐 Secrets Management

### Local Development
Store your API keys in `.streamlit/secrets.toml` (this file is gitignored):

```toml
GEMINI_API_KEY = "your-key-here"
SERPAPI_KEY = "your-key-here"
```

### Streamlit Community Cloud Deployment

1. Go to your app settings on [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Navigate to "Secrets" section
3. Add your secrets in TOML format:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   SERPAPI_KEY = "your-key-here"
   ```

### Using Custom Keys
Users can also provide their own API keys through the sidebar interface without needing to redeploy the app.

## 📊 How It Works

1. **App Search**: Uses SerpApi to search Google Play Store
2. **Review Scraping**: Fetches reviews using `google-play-scraper`
3. **Aspect Extraction**: Gemini AI identifies technical aspects (UI, Performance, Features, etc.)
4. **Sentiment Analysis**: Determines sentiment (Positive/Negative/Neutral) for each aspect
5. **Visualization**: Generates comprehensive charts and insights

## 🎨 Visualizations

- **Word Cloud**: Visual representation of frequently mentioned terms
- **Most Discussed Aspects**: Bar chart of top 10 mentioned aspects
- **Most Loved Aspects**: Positive sentiment breakdown
- **Most Criticized Aspects**: Negative sentiment breakdown
- **Heatmap**: Aspect-sentiment correlation matrix
- **Distribution Charts**: Stacked bar charts showing sentiment percentages

## 📦 Dependencies

```
streamlit
pandas
requests
seaborn
matplotlib
google-generativeai
google-play-scraper
wordcloud
```

## ⚙️ Configuration

Adjust the number of reviews to scrape:
- 10, 20, 50, 100, 250, 500, or 1000 reviews

The app automatically handles rate limiting and retries for API calls.

## 🚨 Important Notes

- **Rate Limits**: The app includes delays between API calls to respect Gemini's rate limits
- **Processing Time**: Analyzing 100 reviews takes ~10-15 minutes due to rate limiting
- **API Costs**: Monitor your Gemini API usage to avoid unexpected charges


## 📝 License

This project is for educational purposes. Please ensure you comply with Google Play Store's Terms of Service and API usage policies.

## 🐛 Troubleshooting

### "No app found" error
- Try using more specific app names
- Use the exact app name as it appears on Play Store

### API Rate Limit errors
- The app has built-in retry logic with delays
- Consider reducing the number of reviews to analyze

### Missing visualizations
- Ensure there are enough reviews with extractable aspects
- Some apps may have reviews in languages other than English

