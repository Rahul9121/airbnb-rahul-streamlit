# 🏙️ NYC Airbnb Data Analysis Dashboard

An interactive web application for exploring Airbnb listings data in New York City, built with Streamlit.

## 📊 Overview

This dashboard provides comprehensive analysis of NYC Airbnb data including:

- **Geographic Distribution**: Interactive maps showing listing locations
- **Pricing Analysis**: Price trends across boroughs and room types
- **Availability Insights**: Room availability and booking patterns
- **Reviews Analytics**: Review patterns and guest feedback analysis
- **Host Analysis**: Host behavior and listing management

## 🚀 Features

- **Interactive Visualizations**: Built with Plotly for dynamic charts and maps
- **Real-time Filtering**: Filter data by borough, room type, price range, etc.
- **Responsive Design**: Works on desktop and mobile devices
- **Professional Styling**: Airbnb-inspired color scheme and design

## 📁 Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── README.md                # Project documentation
└── Airbnb listings EDA/     # Data files directory
    └── CSIT 553_...../
        ├── listings.csv     # Main listings data
        ├── neighbourhoods.csv # Neighborhood data
        └── reviews.csv      # Reviews data
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Development

1. **Clone/Download the project**
   ```bash
   git clone <repository-url>
   cd airbnb-nyc-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **View in browser**
   - Open http://localhost:8501
   - The app will automatically reload when you make changes

## 📊 Data Sources

The dashboard uses three main datasets:

- **listings.csv**: Detailed information about Airbnb listings (39,319 entries)
- **neighbourhoods.csv**: Geographic neighborhood data (230 neighborhoods)
- **reviews.csv**: Historical review data (979,755 reviews)

## 🎯 Key Insights Available

- **Geographic Patterns**: Which neighborhoods have the most listings
- **Pricing Strategies**: Average prices by location and room type
- **Market Dynamics**: Availability and booking patterns
- **Guest Preferences**: Review patterns and satisfaction indicators
- **Host Behavior**: Single vs. multi-listing hosts

## 🛠️ Technical Features

- **Caching**: Uses `@st.cache_data` for fast data loading
- **Performance**: Samples large datasets for responsive maps
- **Responsive**: Mobile-friendly design
- **Interactive**: Real-time filtering and exploration

## 🎨 Customization

You can customize the app by:

1. **Modifying colors** in `.streamlit/config.toml`
2. **Adding new visualizations** in the respective functions
3. **Extending filters** in the sidebar sections
4. **Adding new analysis pages** to the navigation

## 📈 Adding to Your Portfolio

This dashboard demonstrates:

- **Data Science Skills**: EDA, statistical analysis, data visualization
- **Technical Proficiency**: Python, Pandas, Plotly, Streamlit
- **UI/UX Design**: Interactive dashboards, responsive design
- **Deployment**: Cloud deployment and DevOps practices

## 🤝 Contributing

Originally created by:
- **Sai Rohini Godavarthi**
- **Rahul Ega**

For CSIT 553 coursework, converted to interactive dashboard.

## 📝 License

This project is for educational purposes. Data sourced from Airbnb public datasets.


**Live Demo**: https://airbnb-rahul-app-ep3hwhtjgmony9o99rgkqs.streamlit.app/
