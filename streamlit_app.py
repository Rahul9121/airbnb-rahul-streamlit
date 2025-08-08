import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="NYC Airbnb Data Analysis",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF5A5F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 2rem;
        color: #484848;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF5A5F;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        # Load datasets
        df_listings = pd.read_csv('data/listings.csv')
        df_neighbourhoods = pd.read_csv('data/neighbourhoods.csv')
        df_reviews = pd.read_csv('data/reviews.csv')
        
        # Data preprocessing (following the original notebook)
        # Drop columns with high missing values
        df_listings.drop(columns=['last_review', 'license'], inplace=True)
        
        # Fill missing values
        df_listings['reviews_per_month'].fillna(0, inplace=True)
        df_listings['name'].fillna("Unknown", inplace=True)
        df_listings['host_name'].fillna("Unknown", inplace=True)
        
        # Fill price missing values with average by neighbourhood and room type
        avg_prices = df_listings.groupby(['neighbourhood_group', 'room_type'])['price'].mean()
        df_listings['price'] = df_listings.apply(
            lambda row: avg_prices[row['neighbourhood_group'], row['room_type']] 
            if pd.isnull(row['price']) else row['price'], axis=1
        )
        
        return df_listings, df_neighbourhoods, df_reviews
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def main():
    # Title and description
    st.markdown('<h1 class="main-header">🏙️ NYC Airbnb Data Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **An Interactive Exploratory Data Analysis of Airbnb listings in New York City**
    
    This analysis explores detailed information on Airbnb listings, reviews, and neighbourhoods in NYC, 
    providing insights into hospitality trends, guest preferences, and regional accommodation specifics.
    
    *Originally created by Sai Rohini Godavarthi & Rahul Ega for CSIT 553*
    """)
    
    # Load data
    df_listings, df_neighbourhoods, df_reviews = load_data()
    
    if df_listings is None:
        st.error("Failed to load data. Please ensure the CSV files are in the correct directory.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📈 Overview", "🗺️ Geographic Analysis", "💰 Pricing Analysis", 
         "🏠 Room Types & Availability", "⭐ Reviews Analysis", "🔍 Detailed Exploration"]
    )
    
    if page == "📈 Overview":
        show_overview(df_listings, df_neighbourhoods, df_reviews)
    elif page == "🗺️ Geographic Analysis":
        show_geographic_analysis(df_listings)
    elif page == "💰 Pricing Analysis":
        show_pricing_analysis(df_listings)
    elif page == "🏠 Room Types & Availability":
        show_room_availability_analysis(df_listings)
    elif page == "⭐ Reviews Analysis":
        show_reviews_analysis(df_listings, df_reviews)
    elif page == "🔍 Detailed Exploration":
        show_detailed_exploration(df_listings)

def show_overview(df_listings, df_neighbourhoods, df_reviews):
    st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Listings", f"{len(df_listings):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Neighbourhoods", df_listings['neighbourhood'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Reviews", f"{len(df_reviews):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Unique Hosts", df_listings['host_id'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Borough distribution
    st.markdown("### 🏘️ Listings by Borough")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        borough_counts = df_listings['neighbourhood_group'].value_counts()
        fig = px.bar(
            x=borough_counts.index, 
            y=borough_counts.values,
            title="Number of Listings by Borough",
            color=borough_counts.values,
            color_continuous_scale="viridis"
        )
        fig.update_layout(
            xaxis_title="Borough",
            yaxis_title="Number of Listings",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Key Insights:**")
        st.markdown(f"• **{borough_counts.index[0]}** has the most listings ({borough_counts.values[0]:,})")
        st.markdown(f"• **{borough_counts.index[-1]}** has the fewest ({borough_counts.values[-1]:,})")
        st.markdown(f"• Top 2 boroughs account for {(borough_counts.values[:2].sum()/borough_counts.sum()*100):.1f}% of all listings")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Room type distribution
    st.markdown("### 🏠 Room Type Distribution")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        room_counts = df_listings['room_type'].value_counts()
        fig = px.pie(
            values=room_counts.values,
            names=room_counts.index,
            title="Distribution of Room Types"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Room type by borough
        room_borough = df_listings.groupby(['neighbourhood_group', 'room_type']).size().unstack()
        fig = px.bar(
            room_borough,
            title="Room Types by Borough",
            barmode='stack'
        )
        fig.update_layout(
            xaxis_title="Borough",
            yaxis_title="Number of Listings",
            legend_title="Room Type"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_geographic_analysis(df_listings):
    st.markdown('<h2 class="sub-header">🗺️ Geographic Distribution</h2>', unsafe_allow_html=True)
    
    # Interactive map
    st.markdown("### 📍 Airbnb Listings Map")
    
    # Sidebar filters for map
    st.sidebar.markdown("### Map Filters")
    selected_borough = st.sidebar.multiselect(
        "Select Borough(s):",
        options=df_listings['neighbourhood_group'].unique(),
        default=df_listings['neighbourhood_group'].unique()
    )
    
    price_range = st.sidebar.slider(
        "Price Range ($):",
        min_value=int(df_listings['price'].min()),
        max_value=int(df_listings['price'].max()),
        value=(int(df_listings['price'].min()), min(500, int(df_listings['price'].max())))
    )
    
    # Filter data
    filtered_data = df_listings[
        (df_listings['neighbourhood_group'].isin(selected_borough)) &
        (df_listings['price'].between(price_range[0], price_range[1]))
    ]
    
    # Sample data for better performance
    if len(filtered_data) > 5000:
        filtered_data = filtered_data.sample(n=5000, random_state=42)
        st.info(f"Showing a sample of 5,000 listings for better performance. Total matching listings: {len(df_listings[(df_listings['neighbourhood_group'].isin(selected_borough)) & (df_listings['price'].between(price_range[0], price_range[1]))])}")
    
    # Create map
    if not filtered_data.empty:
        fig = px.scatter_mapbox(
            filtered_data,
            lat="latitude",
            lon="longitude",
            color="neighbourhood_group",
            size="price",
            hover_data=["name", "room_type", "price"],
            mapbox_style="open-street-map",
            zoom=10,
            title=f"Airbnb Listings in NYC ({len(filtered_data)} listings shown)"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No listings match the selected filters.")
    
    # Neighborhood analysis
    st.markdown("### 🏘️ Top Neighborhoods by Listings")
    col1, col2 = st.columns(2)
    
    with col1:
        top_neighborhoods = df_listings['neighbourhood'].value_counts().head(15)
        fig = px.bar(
            x=top_neighborhoods.values,
            y=top_neighborhoods.index,
            orientation='h',
            title="Top 15 Neighborhoods by Number of Listings"
        )
        fig.update_layout(
            xaxis_title="Number of Listings",
            yaxis_title="Neighborhood"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average price by top neighborhoods
        avg_price_neighborhood = df_listings.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=avg_price_neighborhood.values,
            y=avg_price_neighborhood.index,
            orientation='h',
            title="Top 15 Neighborhoods by Average Price"
        )
        fig.update_layout(
            xaxis_title="Average Price ($)",
            yaxis_title="Neighborhood"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_pricing_analysis(df_listings):
    st.markdown('<h2 class="sub-header">💰 Pricing Analysis</h2>', unsafe_allow_html=True)
    
    # Price statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Price", f"${df_listings['price'].mean():.2f}")
    with col2:
        st.metric("Median Price", f"${df_listings['price'].median():.2f}")
    with col3:
        st.metric("Min Price", f"${df_listings['price'].min():.2f}")
    with col4:
        st.metric("Max Price", f"${df_listings['price'].max():.2f}")
    
    # Price distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Price Distribution")
        # Remove extreme outliers for better visualization
        price_filtered = df_listings[df_listings['price'] <= df_listings['price'].quantile(0.95)]
        fig = px.histogram(
            price_filtered,
            x="price",
            nbins=50,
            title="Price Distribution (95th percentile filter)"
        )
        fig.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💲 Price by Borough")
        fig = px.box(
            df_listings[df_listings['price'] <= df_listings['price'].quantile(0.95)],
            x="neighbourhood_group",
            y="price",
            title="Price Distribution by Borough"
        )
        fig.update_layout(
            xaxis_title="Borough",
            yaxis_title="Price ($)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Price by room type
    st.markdown("### 🏠 Average Price by Room Type and Borough")
    avg_price_data = df_listings.groupby(['neighbourhood_group', 'room_type'])['price'].mean().unstack()
    
    fig = go.Figure()
    for room_type in avg_price_data.columns:
        fig.add_trace(go.Bar(
            name=room_type,
            x=avg_price_data.index,
            y=avg_price_data[room_type],
            text=avg_price_data[room_type].round(2),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Average Price by Room Type and Borough",
        xaxis_title="Borough",
        yaxis_title="Average Price ($)",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Price insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**💡 Pricing Insights:**")
    most_expensive_borough = df_listings.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False).index[0]
    least_expensive_borough = df_listings.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False).index[-1]
    
    st.markdown(f"• **Most expensive borough:** {most_expensive_borough}")
    st.markdown(f"• **Most affordable borough:** {least_expensive_borough}")
    
    expensive_room_type = df_listings.groupby('room_type')['price'].mean().sort_values(ascending=False).index[0]
    st.markdown(f"• **Most expensive room type:** {expensive_room_type}")
    st.markdown('</div>', unsafe_allow_html=True)

def show_room_availability_analysis(df_listings):
    st.markdown('<h2 class="sub-header">🏠 Room Types & Availability Analysis</h2>', unsafe_allow_html=True)
    
    # Room type metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 Room Type Distribution")
        room_counts = df_listings['room_type'].value_counts()
        fig = px.pie(
            values=room_counts.values,
            names=room_counts.index,
            title="Room Type Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📅 Average Availability")
        avg_availability = df_listings.groupby('room_type')['availability_365'].mean()
        fig = px.bar(
            x=avg_availability.index,
            y=avg_availability.values,
            title="Average Availability by Room Type (days/year)"
        )
        fig.update_layout(
            xaxis_title="Room Type",
            yaxis_title="Average Availability (days)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Availability analysis
    st.markdown("### 📊 Availability Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df_listings,
            x="availability_365",
            color="room_type",
            title="Availability Distribution by Room Type"
        )
        fig.update_layout(
            xaxis_title="Availability (days per year)",
            yaxis_title="Number of Listings"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Minimum nights analysis
        fig = px.box(
            df_listings[df_listings['minimum_nights'] <= 30],  # Filter extreme values
            x="room_type",
            y="minimum_nights",
            title="Minimum Nights Requirement by Room Type"
        )
        fig.update_layout(
            xaxis_title="Room Type",
            yaxis_title="Minimum Nights"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Host listings analysis
    st.markdown("### 👥 Host Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        host_listings = df_listings['calculated_host_listings_count'].value_counts().head(10)
        fig = px.bar(
            x=host_listings.index,
            y=host_listings.values,
            title="Distribution of Host Listing Counts"
        )
        fig.update_layout(
            xaxis_title="Number of Listings per Host",
            yaxis_title="Number of Hosts"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top hosts
        top_hosts = df_listings.groupby('host_name')['calculated_host_listings_count'].first().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=top_hosts.values,
            y=top_hosts.index,
            orientation='h',
            title="Top 10 Hosts by Number of Listings"
        )
        fig.update_layout(
            xaxis_title="Number of Listings",
            yaxis_title="Host Name"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_reviews_analysis(df_listings, df_reviews):
    st.markdown('<h2 class="sub-header">⭐ Reviews Analysis</h2>', unsafe_allow_html=True)
    
    # Reviews metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(df_reviews):,}")
    with col2:
        st.metric("Avg Reviews per Listing", f"{df_listings['number_of_reviews'].mean():.1f}")
    with col3:
        st.metric("Listings with Reviews", f"{(df_listings['number_of_reviews'] > 0).sum():,}")
    with col4:
        st.metric("Review Rate", f"{(df_listings['number_of_reviews'] > 0).mean()*100:.1f}%")
    
    # Reviews distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Reviews Distribution")
        # Filter for better visualization
        reviews_filtered = df_listings[df_listings['number_of_reviews'] <= df_listings['number_of_reviews'].quantile(0.95)]
        fig = px.histogram(
            reviews_filtered,
            x="number_of_reviews",
            nbins=50,
            title="Number of Reviews Distribution"
        )
        fig.update_layout(
            xaxis_title="Number of Reviews",
            yaxis_title="Number of Listings"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏘️ Average Reviews by Borough")
        avg_reviews_borough = df_listings.groupby('neighbourhood_group')['number_of_reviews'].mean()
        fig = px.bar(
            x=avg_reviews_borough.index,
            y=avg_reviews_borough.values,
            title="Average Reviews by Borough"
        )
        fig.update_layout(
            xaxis_title="Borough",
            yaxis_title="Average Number of Reviews"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Reviews timeline
    if 'date' in df_reviews.columns:
        st.markdown("### 📈 Reviews Timeline")
        df_reviews['date'] = pd.to_datetime(df_reviews['date'])
        df_reviews['year_month'] = df_reviews['date'].dt.to_period('M')
        reviews_timeline = df_reviews.groupby('year_month').size()
        
        # Convert Period to string for plotting
        timeline_data = pd.DataFrame({
            'period': reviews_timeline.index.astype(str),
            'count': reviews_timeline.values
        })
        
        fig = px.line(
            timeline_data,
            x='period',
            y='count',
            title="Reviews Over Time"
        )
        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Number of Reviews"
        )
        # Show every 12th label to avoid crowding
        if len(timeline_data) > 12:
            fig.update_xaxes(
                tickmode='array',
                tickvals=timeline_data['period'][::12],
                ticktext=timeline_data['period'][::12]
            )
        st.plotly_chart(fig, use_container_width=True)
    
    # Reviews vs Price correlation
    st.markdown("### 💰 Reviews vs Price Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample data for scatter plot performance
        sample_data = df_listings[df_listings['price'] <= 500].sample(n=min(1000, len(df_listings)), random_state=42)
        fig = px.scatter(
            sample_data,
            x="price",
            y="number_of_reviews",
            color="room_type",
            title="Reviews vs Price (Sample of 1000 listings)"
        )
        fig.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Number of Reviews"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📅 Reviews per Month by Room Type")
        avg_reviews_month = df_listings[df_listings['reviews_per_month'] > 0].groupby('room_type')['reviews_per_month'].mean()
        fig = px.bar(
            x=avg_reviews_month.index,
            y=avg_reviews_month.values,
            title="Average Reviews per Month by Room Type"
        )
        fig.update_layout(
            xaxis_title="Room Type",
            yaxis_title="Average Reviews per Month"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_detailed_exploration(df_listings):
    st.markdown('<h2 class="sub-header">🔍 Detailed Data Exploration</h2>', unsafe_allow_html=True)
    
    # Interactive filters
    st.sidebar.markdown("### 🔧 Data Filters")
    
    selected_boroughs = st.sidebar.multiselect(
        "Select Boroughs:",
        options=df_listings['neighbourhood_group'].unique(),
        default=df_listings['neighbourhood_group'].unique()
    )
    
    selected_room_types = st.sidebar.multiselect(
        "Select Room Types:",
        options=df_listings['room_type'].unique(),
        default=df_listings['room_type'].unique()
    )
    
    price_range = st.sidebar.slider(
        "Price Range ($):",
        min_value=int(df_listings['price'].min()),
        max_value=int(df_listings['price'].max()),
        value=(int(df_listings['price'].min()), min(1000, int(df_listings['price'].max())))
    )
    
    # Filter data
    filtered_df = df_listings[
        (df_listings['neighbourhood_group'].isin(selected_boroughs)) &
        (df_listings['room_type'].isin(selected_room_types)) &
        (df_listings['price'].between(price_range[0], price_range[1]))
    ]
    
    st.markdown(f"### Showing {len(filtered_df):,} listings based on your filters")
    
    # Data table
    if st.checkbox("Show Raw Data"):
        st.dataframe(
            filtered_df[['name', 'neighbourhood_group', 'neighbourhood', 'room_type', 
                        'price', 'minimum_nights', 'number_of_reviews', 'availability_365']].head(100),
            use_container_width=True
        )
    
    # Summary statistics
    st.markdown("### 📊 Summary Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numerical Variables:**")
        st.dataframe(filtered_df[['price', 'minimum_nights', 'number_of_reviews', 
                                'reviews_per_month', 'availability_365']].describe())
    
    with col2:
        st.markdown("**Categorical Variables:**")
        categorical_summary = pd.DataFrame({
            'Borough': filtered_df['neighbourhood_group'].value_counts(),
            'Room Type': filtered_df['room_type'].value_counts()
        }).fillna(0)
        st.dataframe(categorical_summary)
    
    # Correlation analysis
    st.markdown("### 🔗 Correlation Analysis")
    numeric_cols = ['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 
                   'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm']
    
    correlation_matrix = filtered_df[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Correlation Matrix of Numerical Variables",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Custom analysis
    st.markdown("### 🎯 Custom Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Select X variable:", numeric_cols, index=0)
    with col2:
        y_var = st.selectbox("Select Y variable:", numeric_cols, index=1)
    
    if x_var != y_var:
        fig = px.scatter(
            filtered_df.sample(n=min(1000, len(filtered_df)), random_state=42),
            x=x_var,
            y=y_var,
            color="neighbourhood_group",
            title=f"{y_var} vs {x_var}"
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
def show_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888888; padding: 20px;'>
            📊 NYC Airbnb Data Analysis Dashboard<br>
            Originally created by Sai Rohini Godavarthi & Rahul Ega for CSIT 553<br>
            Interactive version powered by Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    show_footer()
