# Windborne Constellation Tracker

A comprehensive web application for real-time tracking, analysis, and operational insights from Windborne Systems' global sounding balloon constellation data.

## Dashboard Preview

https://windborne-dashboard.onrender.com

## Overview

The Windborne Constellation Tracker fetches data from the live Windborne API (`https://a.windbornesystems.com/treasure/XX.json`), processes potentially corrupted data robustly, and provides a feature-rich interface with multiple visualization types, analytical tools and recommendations via llm (claude). The application automatically refreshes data and dynamically updates insights, making it an ideal operational tool for constellation management.

## Features

### Dashboard
- **Summary Statistics**: Active balloons, total balloons, average altitude, average speed
- **Global Distribution Map**: Real-time visualization of balloon positions worldwide
- **Wind Pattern Analysis**: Current wind conditions at different altitude ranges
- **Altitude Distribution**: Histogram of balloon distribution across altitude ranges
- **Balloon Cluster Detection**: Automatic identification of geographic balloon groupings

### Interactive Map
- **Detailed Balloon Tracking**: Position, altitude, speed, and trajectory of each balloon
- **Trajectory Visualization**: Historical paths with direction indicators
- **Filtering Capabilities**: Filter by altitude range or search for specific balloons
- **Cluster Visualization**: Geographic clustering with additional metadata
- **Wind Pattern Overlay**: Optional visualization of wind patterns by altitude

### Analytics
- **Altitude-Speed Correlation**: Analysis of how altitude affects balloon speed
- **Movement Direction Distribution**: Radar chart showing balloon movement patterns
- **Speed Distribution**: Distribution of balloon speeds across the constellation
- **Detailed Data Table**: Sortable, searchable table of all balloon metrics
- **Time-Based Filtering**: Filter data by different time ranges (24h, 12h, 6h)

### Insights
- **Weather Integration**: Combines balloon data with free Open-Meteo API
- **Wind Impact Analysis**: Correlation between wind speeds and balloon performance
- **Altitude-Weather Relationship**: Analysis of optimal altitude ranges based on weather
- **Performance Index**: Custom performance metrics combining speed, distance, and data density
- **Deployment Recommendations**: Actionable insights for optimal balloon deployment
- **Anomaly Detection**: Automatic identification of unusual balloon behavior

### Enhanced Insights
- **Trajectory prediction**: Forecasts balloon positions 6-24 hours ahead
- **Atmospheric anomaly detection**: Compares current conditions with historical averages
- **Performance analytics** Identifies optimal altitude bands for efficient operations
- **Launch site recommendations**: Based on current wind patterns
- **LLM Insights**: Adds AI-powered analysis as an optional feature
- **Enhanced Visualization**: Interactive trajectory map with confidence indicators, 
                              Atmospheric anomaly detection with explanation, capabilities,
                              Optimal altitude band recommendations with performance metrics, and
                              Efficiency metrics with visual indicators 

## Project Structure

```
windborne-constellation/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI main entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py    # API endpoints
|   |── data/
│   │   ├── weather_cache.pkl
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_fetcher.py  # Fetch and clean data from Windborne API
|   |   ├── llm_insights_service.py # Generate insights using Claude
|   |   ├── weather_data_service.py # Fetch & process weather data from open-meteo
│   │   └── data_processor.py # Process and analyze data
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css    # Custom styling
│   │   └── js/
│   │       └── utils.js     # Utility functions
│   └── templates/        # HTML templates
│       ├── base.html     # Base template
│       ├── index.html    # Dashboard page
│       ├── map_view.html # Interactive map page
│       ├── analytics.html # Detailed analytics page
│       ├── enhanced_insights .html # AI based insights page
│       └── insights.html # Operational insights page
├── .gitignore
├── LICENSE
├── procfile
├── render.yaml
├── requirements.txt     # Python dependencies
├── README.md
└── run.py              # Local development entry point
```

## Technical Implementation

### Backend (Python/FastAPI)
- **API Handling**: Robust API client with error handling, retries, and data cleaning
- **Data Processing**: Comprehensive analysis including trajectory calculations, clustering, and anomaly detection
- **Caching System**: Intelligent caching to minimize API calls while keeping data fresh
- **Asynchronous Processing**: Background tasks for data refresh without blocking user interactions

### Frontend (HTML/JavaScript)
- **Interactive Maps**: Leaflet.js for geographic visualizations
- **Data Visualization**: Chart.js for interactive charts and graphs
- **Responsive Design**: Bootstrap 5 framework for responsive layouts
- **Real-time Updates**: Periodic data refresh using JavaScript

### Data Analysis Algorithms
- **Proximity-based Tracking**: It identifies the same physical balloon across time periods by finding the nearest balloon in subsequent hours, which better represents actual balloon movement.
- **Haversine Formula**: Accurate distance calculations on Earth's surface
- **Balloon Clustering**: Geographic clustering algorithm to identify balloon groupings
- **Wind Pattern Analysis**: Vector-based analysis of wind directions and speeds
- **Performance Index**: Custom algorithm combining multiple metrics to evaluate balloon performance
- **Anomaly Detection**: Statistical outlier detection across multiple parameters

### Performance Optimizations

- **Centralized Utility Functions**: Common geographic calculations moved to a utility module
- **Request Timeouts**: Implemented for all external API calls
- **Loading Indicators**: Clear visual feedback during data processing
- **Deferred API Validation**: Optional API keys only validated on demand, not on page load
- **Optimized Data Caching**: Minimized redundant API calls with intelligent caching

### Optional AI Integration
- **Claude API Integration**: Optional integration with Anthropic's Claude API
- **Client-Side Security**: API keys stored locally in the browser, never sent to our servers
- **Request Timeouts**: Implemented to prevent UI freezes during API calls
- **Graceful Fallbacks**: System functions without AI if API key isn't provided

## Metrics Explained

### Balloon-Specific Metrics
- **Average Speed**: Mean velocity of each balloon over its recorded history (km/h)
- **Total Distance**: Cumulative distance traveled by each balloon (km)
- **Direction**: Cardinal or intercardinal direction of balloon movement
- **Altitude Change**: Net change in altitude over recorded history (m)
- **Performance Index**: Composite score (0-100) based on speed, distance, and data density

### Constellation-Wide Metrics
- **Active Balloons**: Number of balloons with recent data updates (within 2 hours)
- **Altitude Distribution**: Statistical breakdown of balloons across altitude ranges
- **Wind Patterns**: Average wind direction and speed at different altitude ranges
- **Cluster Analysis**: Number and size of geographic balloon groupings
- **Speed Distribution**: Statistical breakdown of balloon speeds across the constellation

## APIs Used

### Internal APIs
- `GET /api/balloons`: Returns all balloon data with trajectory history
- `GET /api/enhanced-balloon-data`: Returns enhanced balloon data with predictions
- `GET /api/weather-data`: Returns processed weather data from Open-Meteo and OpenWeather
- `GET /api/refresh`: Forces a refresh of the constellation data
- `POST /api/validate-anthropic-key`: Validates Anthropic API keys
- `POST /api/generate-insights`: Generates AI-powered constellation insights
- `POST /api/explain-anomaly`: Generates explanations for detected anomalies
- `POST /api/trajectory-analysis`: Analyzes balloon trajectory predictions

### External APIs
- **Windborne Systems API**: `https://a.windbornesystems.com/treasure/XX.json` (where XX is hours from 00-23)
- **Open-Meteo API**: Used for weather data (`https://api.open-meteo.com/v1/forecast`)
- **OpenWeather API** (optional): Used for additional weather visualizations and overlays (`https://api.openweathermap.org`)
- **Anthropic Claude API** (optional): Used for AI-powered analysis and insights
- **OpenStreetMap**: Base map tiles for geographic visualizations

## Setup and Deployment

### Local Development

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/windborne-dashboard.git
   cd windborne-dashboard
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python run.py
   ```

5. Open your browser and navigate to `http://localhost:8000`

## Troubleshooting

### Common Issues

1. **Data Not Loading**:
   - Check your internet connection
   - Verify the Windborne API is accessible
   - Check browser console for JavaScript errors

2. **Maps Not Displaying**:
   - Ensure Leaflet.js is loading correctly
   - Check if your browser blocks third-party map tiles

3. **Enhanced Insights Page Loading Slowly**:
   - Use the "Apply" button to validate your API key instead of relying on automatic validation
   - Check browser console for timeout errors
   - Ensure your Anthropic API key starts with 'sk-ant-'

3. **Weather Overlay Not Working**:
   - Verify your OpenWeather API key is correct
   - Check browser console for API-related errors
   - Ensure your internet connection is stable

### Support

For issues, questions, or contributions, please:
- Open an issue on GitHub

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Windborne Systems](https://windbornesystems.com) for the constellation API
- [OpenMeteo](https://open-meteo.com) for weather data
- [OpenWeather](https://openweathermap.org) for additional weather data and visualizations
- [OpenStreetMap](https://www.openstreetmap.org) for map tiles
- [Leaflet.js](https://leafletjs.com) for interactive maps
- [Chart.js](https://www.chartjs.org) for data visualizations
- [Bootstrap](https://getbootstrap.com) for responsive design
- [FastAPI](https://fastapi.tiangolo.com) for the backend framework
- [Anthropic](https://www.anthropic.com) for Claude API
