# LiftOS Streamlit Hub

A modern, Python-based web interface for the LiftOS Marketing Intelligence Platform. This Streamlit application provides an intuitive hub for accessing all LiftOS microservices through a unified, data science-friendly interface.

## 🚀 Features

### 🧠 Causal Analysis
- **Platform Data Sync**: One-click integration with Meta Ads, Google Ads, and Klaviyo
- **Attribution Modeling**: Multiple attribution models (first-touch, last-touch, time-decay, etc.)
- **Budget Optimization**: AI-powered budget allocation recommendations
- **Lift Experiments**: A/B testing and incremental lift measurement

### 🔍 Memory Search
- **Semantic Search**: Natural language search across all your marketing data
- **Smart Filtering**: Filter by date, tags, and data types
- **Saved Queries**: Save and reuse common searches
- **Recent Activity**: Track all platform activities and analyses

### 🤖 LLM Assistant
- **AI-Powered Insights**: Ask questions about your marketing data in natural language
- **Contextual Responses**: AI understands your current data and recent activities
- **Suggested Questions**: Pre-built questions to get you started
- **Chat History**: Full conversation history with export capabilities

### 📊 Interactive Visualizations
- **Real-time Charts**: Plotly-powered interactive visualizations
- **Attribution Charts**: Channel performance and attribution breakdowns
- **Budget Allocation**: Pie charts and bar charts for budget optimization
- **Lift Measurement**: Treatment vs control group comparisons

## 🛠️ Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional)
- LiftOS microservices running (or demo mode)

### Local Development

1. **Clone and Setup**
   ```bash
   cd liftos-streamlit
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

4. **Access the Hub**
   Open your browser to `http://localhost:8501`

### Docker Deployment

1. **Build and Run**
   ```bash
   cd docker
   docker-compose up -d
   ```

2. **Access the Hub**
   Open your browser to `http://localhost:8501`

## 📋 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `DEMO_MODE` | `true` | Enable demo mode with mock data |
| `REQUIRE_AUTH` | `false` | Require user authentication |
| `CAUSAL_SERVICE_URL` | `http://localhost:8001` | Causal microservice URL |
| `MEMORY_SERVICE_URL` | `http://localhost:8004` | Memory microservice URL |
| `LLM_SERVICE_URL` | `http://localhost:8003` | LLM microservice URL |

### Feature Flags

Enable/disable specific features:
- `ENABLE_CAUSAL=true` - Causal analysis features
- `ENABLE_LLM=true` - LLM assistant
- `ENABLE_MEMORY=true` - Memory search
- `ENABLE_EXPERIMENTS=true` - Lift experiments

## 🎯 Usage Guide

### Getting Started

1. **Demo Mode**: The application starts in demo mode by default - no authentication required
2. **Navigation**: Use the sidebar to navigate between different modules
3. **Data Sync**: Start by syncing data from your marketing platforms
4. **Analysis**: Run attribution analysis and budget optimization
5. **Ask Questions**: Use the LLM assistant for insights and recommendations

### Causal Analysis Workflow

1. **Sync Platform Data**
   - Click "Meta", "Google", or "Klaviyo" sync buttons
   - Configure date ranges and attribution windows
   - Review synced data in the Data Overview tab

2. **Run Attribution Analysis**
   - Select attribution model (time-decay recommended)
   - Set attribution window (14 days default)
   - Click "Run Attribution Analysis"
   - Review results and download charts

3. **Optimize Budget**
   - Set total budget and optimization goal
   - Run budget optimization
   - Review recommended allocations
   - Export optimization results

4. **Measure Lift**
   - Create lift experiments
   - Monitor experiment progress
   - Analyze treatment vs control results

### Memory Search

1. **Natural Language Search**
   - Type questions like "Meta ads performance last month"
   - Use advanced filters for precise results
   - Save useful queries for later

2. **Browse Recent Activity**
   - View all recent analyses and syncs
   - Click on activities for detailed views
   - Export activity reports

### LLM Assistant

1. **Ask Questions**
   - Type questions about your marketing data
   - Get AI-powered insights and recommendations
   - Use suggested questions to get started

2. **Context-Aware Responses**
   - AI understands your current data state
   - Provides personalized recommendations
   - References your recent activities

## 🏗️ Architecture

### Component Structure
```
liftos-streamlit/
├── app.py                    # Main application
├── config/                   # Configuration management
├── auth/                     # Authentication system
├── pages/                    # Streamlit pages
├── components/               # Reusable UI components
├── utils/                    # Utilities and API clients
└── docker/                   # Docker configuration
```

### Microservice Integration
- **API Client**: Centralized HTTP client for all microservices
- **Error Handling**: Graceful fallbacks and user-friendly error messages
- **Caching**: Smart caching for improved performance
- **Authentication**: Token-based auth with session management

## 🔧 Development

### Adding New Pages

1. Create a new file in `pages/` following the naming convention: `N_🔥_Page_Name.py`
2. Import required modules and follow the existing page structure
3. Add navigation button in `components/sidebar.py`
4. Update feature flags in `config/settings.py` if needed

### Adding New Charts

1. Add chart function to `components/charts.py`
2. Follow Plotly conventions for consistency
3. Include download functionality
4. Add responsive design considerations

### API Integration

1. Add new methods to `utils/api_client.py`
2. Include proper error handling
3. Add caching for expensive operations
4. Provide mock data for demo mode

## 🧪 Testing

### Manual Testing
```bash
# Run the application
streamlit run app.py

# Test different scenarios:
# 1. Demo mode (default)
# 2. With microservices running
# 3. With authentication enabled
# 4. Different feature flag combinations
```

### Docker Testing
```bash
# Build and test
cd docker
docker-compose up --build

# Test health endpoints
curl http://localhost:8501/_stcore/health
```

## 📊 Performance

### Optimization Features
- **Caching**: API responses cached for 5 minutes by default
- **Lazy Loading**: Components load only when needed
- **Efficient Rendering**: Minimal re-renders with proper state management
- **Memory Management**: Automatic cleanup of old session data

### Monitoring
- Health checks for all services
- Performance metrics in sidebar
- Error tracking and logging
- User activity monitoring

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   # Set production environment variables
   export ENVIRONMENT=production
   export DEBUG=false
   export REQUIRE_AUTH=true
   ```

2. **Docker Deployment**
   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```

3. **Health Verification**
   ```bash
   curl http://your-domain:8501/_stcore/health
   ```

### Scaling Considerations
- **Horizontal Scaling**: Multiple Streamlit instances behind load balancer
- **Session Persistence**: Use external session store for multi-instance deployments
- **Caching**: Consider Redis for shared caching across instances
- **Database**: External database for persistent storage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings for all functions
- Keep functions focused and small

## 📄 License

This project is part of the LiftOS platform. See the main LiftOS repository for license information.

## 🆘 Support

- **Documentation**: See the main LiftOS documentation
- **Issues**: Report bugs in the main LiftOS repository
- **Community**: Join the LiftOS community discussions
- **Enterprise**: Contact support@liftos.ai for enterprise support

---

**Built with ❤️ using Streamlit and the LiftOS platform**