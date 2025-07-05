import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_client import APIClient
from components.sidebar import render_sidebar
from auth.authenticator import authenticate_user
from auth.session_manager import initialize_session

st.set_page_config(page_title="Memory Search", page_icon="🧠", layout="wide")

def main():
    """Main memory search page"""
    
    # Initialize session and authenticate
    initialize_session()
    if not authenticate_user():
        st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("🧠 Memory Search")
    st.markdown("### Search and Explore Your LiftOS Data")
    
    # Initialize API client
    api_client = APIClient()
    
    # Search interface
    render_search_interface(api_client)
    
    # Recent searches
    render_recent_searches()
    
    # Saved queries
    render_saved_queries()

def render_search_interface(api_client: APIClient):
    """Render the main search interface"""
    
    st.subheader("🔍 Search Your Data")
    
    # Search form
    with st.form("memory_search_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search Query",
                placeholder="e.g., 'Meta ads performance last month' or 'attribution models with ROAS > 3'",
                help="Use natural language to search your data"
            )
        
        with col2:
            search_button = st.form_submit_button("🔍 Search", use_container_width=True)
    
    # Advanced search options
    with st.expander("🔧 Advanced Search Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_type = st.selectbox(
                "Search Type",
                ["semantic", "keyword", "hybrid"],
                help="Type of search to perform"
            )
        
        with col2:
            date_filter = st.date_input(
                "Date Range",
                value=[],
                help="Filter results by date range"
            )
        
        with col3:
            result_limit = st.number_input(
                "Max Results",
                min_value=5,
                max_value=100,
                value=20,
                help="Maximum number of results to return"
            )
        
        # Tag filters
        available_tags = ["attribution", "budget", "optimization", "experiment", "sync", "model"]
        selected_tags = st.multiselect(
            "Filter by Tags",
            available_tags,
            help="Filter results by specific tags"
        )
    
    # Perform search
    if search_button and search_query:
        perform_search(api_client, search_query, search_type, date_filter, selected_tags, result_limit)

def perform_search(api_client: APIClient, query: str, search_type: str, date_filter, tags, limit: int):
    """Perform memory search"""
    
    with st.spinner("🔍 Searching your data..."):
        try:
            # Prepare search filters
            filters = {
                "search_type": search_type,
                "limit": limit
            }
            
            if date_filter and len(date_filter) == 2:
                filters["start_date"] = str(date_filter[0])
                filters["end_date"] = str(date_filter[1])
            
            if tags:
                filters["tags"] = tags
            
            # Perform search
            results = api_client.search_memory(query, filters)
            
            # Store results in session state
            st.session_state.search_results = results
            st.session_state.last_search_query = query
            
            # Add to recent searches
            add_to_recent_searches(query)
            
            st.success(f"✅ Found {len(results.get('results', []))} results")
            
            # Display results
            display_search_results(results)
            
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")
            
            # Show mock results for demo
            if "demo" in st.session_state.get('username', '').lower():
                st.info("🎭 Showing demo search results...")
                mock_results = generate_mock_search_results(query)
                st.session_state.search_results = mock_results
                display_search_results(mock_results)

def display_search_results(results: dict):
    """Display search results"""
    
    if not results or 'results' not in results:
        st.info("No results found for your search query.")
        return
    
    st.subheader(f"📊 Search Results ({len(results['results'])} found)")
    
    # Results summary
    if 'summary' in results:
        with st.expander("📋 Search Summary", expanded=True):
            st.markdown(results['summary'])
    
    # Individual results
    for i, result in enumerate(results['results']):
        with st.expander(f"📄 {result.get('title', f'Result {i+1}')}", expanded=i < 3):
            
            # Result metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Relevance Score", f"{result.get('score', 0):.2f}")
            with col2:
                st.metric("Type", result.get('type', 'Unknown'))
            with col3:
                st.metric("Date", result.get('date', 'Unknown'))
            
            # Result content
            if 'content' in result:
                st.markdown("**Content:**")
                if isinstance(result['content'], dict):
                    st.json(result['content'])
                else:
                    st.markdown(result['content'])
            
            # Tags
            if 'tags' in result and result['tags']:
                st.markdown("**Tags:**")
                tag_cols = st.columns(len(result['tags']))
                for j, tag in enumerate(result['tags']):
                    with tag_cols[j]:
                        st.badge(tag)
            
            # Actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"📋 Copy", key=f"copy_{i}"):
                    st.write("Content copied to clipboard!")
            with col2:
                if st.button(f"💾 Save", key=f"save_{i}"):
                    save_result_to_favorites(result)
            with col3:
                if st.button(f"🔗 View Details", key=f"details_{i}"):
                    show_result_details(result)

def render_recent_searches():
    """Render recent searches section"""
    
    st.subheader("🕒 Recent Searches")
    
    recent_searches = st.session_state.get('recent_searches', [])
    
    if not recent_searches:
        st.info("No recent searches. Start searching to see your history here.")
        return
    
    # Display recent searches as clickable buttons
    cols = st.columns(min(len(recent_searches), 3))
    
    for i, search in enumerate(recent_searches[-6:]):  # Show last 6 searches
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(f"🔍 {search}", key=f"recent_{i}", use_container_width=True):
                # Re-run the search
                st.session_state.search_query_input = search
                st.rerun()

def render_saved_queries():
    """Render saved queries section"""
    
    st.subheader("💾 Saved Queries")
    
    saved_queries = st.session_state.get('saved_queries', [])
    
    if not saved_queries:
        st.info("No saved queries yet. Save useful searches for quick access.")
        return
    
    # Display saved queries
    for i, query in enumerate(saved_queries):
        with st.expander(f"📌 {query['name']}", expanded=False):
            st.markdown(f"**Query:** {query['query']}")
            st.markdown(f"**Created:** {query['created_date']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"🔍 Run", key=f"run_saved_{i}"):
                    st.session_state.search_query_input = query['query']
                    st.rerun()
            with col2:
                if st.button(f"✏️ Edit", key=f"edit_saved_{i}"):
                    edit_saved_query(i, query)
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_saved_{i}"):
                    delete_saved_query(i)

def add_to_recent_searches(query: str):
    """Add query to recent searches"""
    if 'recent_searches' not in st.session_state:
        st.session_state.recent_searches = []
    
    # Remove if already exists
    if query in st.session_state.recent_searches:
        st.session_state.recent_searches.remove(query)
    
    # Add to beginning
    st.session_state.recent_searches.insert(0, query)
    
    # Keep only last 10
    st.session_state.recent_searches = st.session_state.recent_searches[:10]

def save_result_to_favorites(result: dict):
    """Save a search result to favorites"""
    if 'favorite_results' not in st.session_state:
        st.session_state.favorite_results = []
    
    # Add timestamp
    result['saved_at'] = pd.Timestamp.now().isoformat()
    
    st.session_state.favorite_results.append(result)
    st.success("✅ Result saved to favorites!")

def show_result_details(result: dict):
    """Show detailed view of a result"""
    st.subheader(f"📄 {result.get('title', 'Result Details')}")
    
    # Full content display
    if 'content' in result:
        if isinstance(result['content'], dict):
            # If it's structured data, show as expandable sections
            for key, value in result['content'].items():
                with st.expander(f"📊 {key.title()}", expanded=True):
                    if isinstance(value, (list, dict)):
                        st.json(value)
                    else:
                        st.write(value)
        else:
            st.markdown(result['content'])
    
    # Metadata
    with st.expander("ℹ️ Metadata", expanded=True):
        metadata = {
            "ID": result.get('id', 'Unknown'),
            "Type": result.get('type', 'Unknown'),
            "Score": result.get('score', 0),
            "Date": result.get('date', 'Unknown'),
            "Tags": ', '.join(result.get('tags', []))
        }
        
        for key, value in metadata.items():
            st.write(f"**{key}:** {value}")

def edit_saved_query(index: int, query: dict):
    """Edit a saved query"""
    st.subheader("✏️ Edit Saved Query")
    
    with st.form("edit_query_form"):
        new_name = st.text_input("Query Name", value=query['name'])
        new_query = st.text_area("Query", value=query['query'])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("💾 Save Changes"):
                st.session_state.saved_queries[index] = {
                    'name': new_name,
                    'query': new_query,
                    'created_date': query['created_date'],
                    'modified_date': pd.Timestamp.now().isoformat()
                }
                st.success("✅ Query updated!")
                st.rerun()
        
        with col2:
            if st.form_submit_button("❌ Cancel"):
                st.rerun()

def delete_saved_query(index: int):
    """Delete a saved query"""
    if 'saved_queries' in st.session_state:
        del st.session_state.saved_queries[index]
        st.success("✅ Query deleted!")
        st.rerun()

def generate_mock_search_results(query: str) -> dict:
    """Generate mock search results for demo"""
    
    # Simulate different types of results based on query
    results = []
    
    if "meta" in query.lower() or "facebook" in query.lower():
        results.append({
            "id": "meta_001",
            "title": "Meta Ads Campaign Performance - December 2024",
            "type": "attribution_analysis",
            "score": 0.95,
            "date": "2024-12-15",
            "content": {
                "campaign_name": "Holiday Sale Campaign",
                "spend": 15000,
                "conversions": 450,
                "roas": 3.2,
                "attribution_value": 48000
            },
            "tags": ["meta", "attribution", "campaign"]
        })
    
    if "attribution" in query.lower():
        results.append({
            "id": "attr_001",
            "title": "Time Decay Attribution Model Results",
            "type": "model_results",
            "score": 0.88,
            "date": "2024-12-20",
            "content": {
                "model_type": "time_decay",
                "channels": ["Meta", "Google", "Klaviyo"],
                "total_attribution": 125000,
                "top_channel": "Meta Ads"
            },
            "tags": ["attribution", "model", "analysis"]
        })
    
    if "budget" in query.lower() or "optimization" in query.lower():
        results.append({
            "id": "budget_001",
            "title": "Budget Optimization Results - Q4 2024",
            "type": "optimization",
            "score": 0.82,
            "date": "2024-12-18",
            "content": {
                "total_budget": 50000,
                "recommended_allocation": {
                    "Meta": 25000,
                    "Google": 18000,
                    "Klaviyo": 7000
                },
                "expected_lift": "15%"
            },
            "tags": ["budget", "optimization", "allocation"]
        })
    
    # Add some general results if no specific matches
    if not results:
        results = [
            {
                "id": "general_001",
                "title": "Recent Platform Sync - All Channels",
                "type": "data_sync",
                "score": 0.75,
                "date": "2025-01-04",
                "content": "Successfully synced data from Meta, Google, and Klaviyo platforms. Total records: 15,420",
                "tags": ["sync", "data", "platforms"]
            },
            {
                "id": "general_002",
                "title": "Lift Experiment Results",
                "type": "experiment",
                "score": 0.70,
                "date": "2025-01-03",
                "content": {
                    "experiment_name": "Meta Ads Lift Test",
                    "lift_percentage": 12.5,
                    "confidence": 95,
                    "status": "completed"
                },
                "tags": ["experiment", "lift", "meta"]
            }
        ]
    
    return {
        "results": results,
        "total_count": len(results),
        "query": query,
        "search_time": 0.15,
        "summary": f"Found {len(results)} relevant results for '{query}'. Results include attribution analyses, campaign data, and optimization insights."
    }

if __name__ == "__main__":
    main()