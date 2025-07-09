import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_client import APIClient
from components.sidebar import render_sidebar
from auth.authenticator import authenticate_user
from auth.session_manager import initialize_session

st.set_page_config(page_title="LLM", page_icon="🤖", layout="wide")

def main():
    """Main LLM Assistant page"""
    
    # Initialize session and authenticate
    initialize_session()
    if not authenticate_user():
        st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("🤖 LLM")
    st.markdown("### Your AI-Powered Marketing Intelligence Assistant")
    
    # Initialize API client
    api_client = APIClient()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Render chat interface
    render_chat_interface(api_client)
    
    # Render suggested questions
    render_suggested_questions(api_client)

def render_chat_interface(api_client: APIClient):
    """Render the main chat interface"""
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message['content'])
                    
                    # Add action buttons for assistant responses
                    if i == len(st.session_state.chat_history) - 1:  # Only for the last response
                        col1, col2, col3 = st.columns([1, 1, 4])
                        with col1:
                            if st.button("👍", key=f"like_{i}", help="Helpful response"):
                                st.success("Thanks for the feedback!")
                        with col2:
                            if st.button("👎", key=f"dislike_{i}", help="Not helpful"):
                                st.info("We'll improve our responses!")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your marketing data..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = get_ai_response(api_client, prompt)
                st.markdown(response)
                
                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})

def get_ai_response(api_client: APIClient, question: str) -> str:
    """Get response from LLM assistant"""
    
    try:
        # Prepare context from recent data
        context = prepare_context_for_llm()
        
        # Call LLM service
        result = api_client.ask_llm(question, context)
        
        return result.get('response', 'I apologize, but I encountered an issue processing your request.')
        
    except Exception as e:
        # Fallback to mock responses for demo
        if "demo" in st.session_state.get('username', '').lower():
            return generate_mock_response(question)
        else:
            return f"I'm sorry, I encountered an error: {str(e)}. Please try again or contact support if the issue persists."

def prepare_context_for_llm() -> str:
    """Prepare context from user's recent data and activities"""
    
    context_parts = []
    
    # Add recent attribution results
    if 'attribution_results' in st.session_state:
        context_parts.append("Recent Attribution Analysis: User has run attribution analysis with results showing channel performance.")
    
    # Add synced platform data
    synced_platforms = []
    for platform in ['meta', 'google', 'klaviyo']:
        if f'{platform}_data' in st.session_state:
            synced_platforms.append(platform)
    
    if synced_platforms:
        context_parts.append(f"Synced Platforms: User has data from {', '.join(synced_platforms)}.")
    
    # Add recent experiments
    if 'running_experiments' in st.session_state and st.session_state.running_experiments:
        exp_count = len(st.session_state.running_experiments)
        context_parts.append(f"Active Experiments: User has {exp_count} running experiments.")
    
    # Add budget optimization results
    if 'budget_optimization_results' in st.session_state:
        context_parts.append("Budget Optimization: User has recent budget optimization results.")
    
    return " ".join(context_parts) if context_parts else "User is new to the platform."

def generate_mock_response(question: str) -> str:
    """Generate mock AI responses for demo purposes"""
    
    question_lower = question.lower()
    
    # Attribution-related questions
    if any(word in question_lower for word in ['attribution', 'channel', 'performance']):
        return """Based on your recent attribution analysis, here are the key insights:

📊 **Channel Performance:**
- Meta Ads is your top performer with $25,000 in attribution value
- Google Ads follows with $18,000 attribution value
- Klaviyo contributes $12,000 in attribution value

💡 **Recommendations:**
1. **Increase Meta Ads budget** - It's showing the highest attribution value
2. **Optimize Google Ads targeting** - Good volume but room for efficiency improvement
3. **Enhance Klaviyo segmentation** - Email marketing shows solid performance

🎯 **Next Steps:**
- Run a lift experiment on Meta Ads to validate incremental impact
- Consider budget reallocation based on attribution efficiency
- Set up automated attribution tracking for ongoing optimization"""

    # Budget-related questions
    elif any(word in question_lower for word in ['budget', 'spend', 'allocation', 'optimize']):
        return """Here's my analysis of your budget optimization opportunities:

💰 **Current Budget Analysis:**
- Total budget: $50,000 across 4 channels
- Meta Ads: $20,000 (40%) - **Underallocated**
- Google Ads: $15,000 (30%) - **Optimal**
- Klaviyo: $8,000 (16%) - **Slightly overallocated**
- TikTok: $7,000 (14%) - **Overallocated**

🎯 **Recommended Reallocation:**
- **Increase Meta Ads to $25,000** (+$5,000) - Highest ROAS
- **Maintain Google Ads at $18,000** (+$3,000) - Steady performer
- **Reduce Klaviyo to $5,000** (-$3,000) - Optimize efficiency
- **Reduce TikTok to $2,000** (-$5,000) - Lowest attribution

📈 **Expected Impact:**
- **15% increase in overall ROAS**
- **$12,000 additional attribution value**
- **Better channel efficiency across the board**"""

    # Data-related questions
    elif any(word in question_lower for word in ['data', 'sync', 'platform', 'connect']):
        return """Here's the status of your data connections:

🔗 **Connected Platforms:**
- ✅ **Meta Ads** - Last synced today, 1,250 records
- ✅ **Google Ads** - Last synced today, 980 records  
- ✅ **Klaviyo** - Last synced today, 2,100 records

📊 **Data Quality:**
- **Completeness**: 98% - Excellent data coverage
- **Freshness**: Real-time sync enabled
- **Accuracy**: No data quality issues detected

🔧 **Recommendations:**
1. **Set up automated daily syncs** for all platforms
2. **Enable real-time conversion tracking** for better attribution
3. **Add UTM parameter standardization** across campaigns
4. **Consider connecting TikTok Ads** for complete view

⚡ **Quick Actions:**
- Sync data now using the sidebar controls
- Review data quality in the Data Overview tab
- Set up alerts for sync failures"""

    # Experiment-related questions
    elif any(word in question_lower for word in ['experiment', 'test', 'lift', 'ab test']):
        return """Let me help you with experiment design and analysis:

🧪 **Current Experiments:**
- You have active lift experiments running
- Meta Ads lift test showing promising early results
- 95% statistical confidence achieved

📈 **Experiment Insights:**
- **Incremental lift**: 12.5% above baseline
- **Test duration**: 14 days (optimal)
- **Sample size**: Sufficient for reliable results

🎯 **Recommendations for New Experiments:**
1. **Google Ads Audience Test** - Test broad vs. narrow targeting
2. **Klaviyo Send Time Optimization** - Test morning vs. evening sends
3. **Cross-Channel Attribution Test** - Measure interaction effects

⚙️ **Best Practices:**
- Run experiments for at least 2 weeks
- Ensure 80% statistical power
- Use holdout groups for clean measurement
- Document all experiment parameters"""

    # General help
    elif any(word in question_lower for word in ['help', 'how', 'what', 'explain']):
        return """I'm here to help you get the most out of LiftOS! Here's what I can assist with:

🧠 **Causal Analysis:**
- Interpret attribution results
- Recommend model improvements
- Explain statistical significance

💰 **Budget Optimization:**
- Analyze spend efficiency
- Suggest reallocation strategies
- Calculate ROI projections

🔍 **Data Insights:**
- Identify trends and patterns
- Explain performance changes
- Recommend data collection improvements

🧪 **Experiment Design:**
- Help design A/B tests
- Interpret experiment results
- Suggest follow-up experiments

💡 **Quick Tips:**
- Ask specific questions about your data
- Request explanations of any metrics
- Get recommendations for next steps
- Learn about advanced features

Try asking: "Why is my Meta Ads performance declining?" or "How should I allocate my budget next month?"
"""

    # Default response
    else:
        return """I understand you're asking about your marketing data. Let me help you with that!

Based on your current LiftOS setup, I can assist with:

🎯 **Your Data:** You have connected Meta Ads, Google Ads, and Klaviyo
📊 **Analysis:** Attribution models and performance insights
💰 **Optimization:** Budget allocation and channel efficiency
🧪 **Testing:** Experiment design and lift measurement

Could you be more specific about what you'd like to know? For example:
- "How is my Meta Ads performing?"
- "Should I increase my Google Ads budget?"
- "What's my best performing channel?"
- "How do I improve my ROAS?"

I'm here to help you make data-driven marketing decisions! 🚀"""

def render_suggested_questions(api_client: APIClient):
    """Render suggested questions section"""
    
    st.markdown("---")
    st.subheader("💡 Suggested Questions")
    st.markdown("Click on any question to get started:")
    
    # Organize suggestions by category
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Performance Analysis**")
        suggestions_performance = [
            "How is my Meta Ads performing this month?",
            "Which channel has the best ROAS?",
            "Why did my conversions drop last week?",
            "What's my customer acquisition cost by channel?"
        ]
        
        for suggestion in suggestions_performance:
            if st.button(suggestion, key=f"perf_{suggestion[:20]}", use_container_width=True):
                ask_suggested_question(suggestion)
    
    with col2:
        st.markdown("**💰 Budget & Optimization**")
        suggestions_budget = [
            "How should I allocate my budget next month?",
            "Which channels should I increase spending on?",
            "What's the optimal attribution window?",
            "How can I improve my overall ROAS?"
        ]
        
        for suggestion in suggestions_budget:
            if st.button(suggestion, key=f"budget_{suggestion[:20]}", use_container_width=True):
                ask_suggested_question(suggestion)
    
    # Additional categories
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**🧪 Experiments & Testing**")
        suggestions_experiments = [
            "How do I design a lift experiment?",
            "What's the statistical significance of my test?",
            "Should I run more A/B tests?",
            "How long should I run my experiment?"
        ]
        
        for suggestion in suggestions_experiments:
            if st.button(suggestion, key=f"exp_{suggestion[:20]}", use_container_width=True):
                ask_suggested_question(suggestion)
    
    with col4:
        st.markdown("**🔍 Data & Insights**")
        suggestions_data = [
            "What trends do you see in my data?",
            "How often should I sync my platforms?",
            "What data quality issues should I watch for?",
            "How do I interpret attribution results?"
        ]
        
        for suggestion in suggestions_data:
            if st.button(suggestion, key=f"data_{suggestion[:20]}", use_container_width=True):
                ask_suggested_question(suggestion)

def ask_suggested_question(question: str):
    """Handle suggested question click"""
    # Add to chat history and trigger response
    st.session_state.chat_history.append({"role": "user", "content": question})
    
    # Get AI response
    api_client = APIClient()
    response = get_ai_response(api_client, question)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Rerun to show the new messages
    st.rerun()

# Sidebar additions for LLM page
def render_llm_sidebar():
    """Render LLM-specific sidebar content"""
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("🤖 Chat Options")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        # Export chat
        if st.button("📥 Export Chat", use_container_width=True):
            if st.session_state.chat_history:
                chat_text = "\n\n".join([
                    f"**{msg['role'].title()}:** {msg['content']}" 
                    for msg in st.session_state.chat_history
                ])
                st.download_button(
                    label="Download Chat History",
                    data=chat_text,
                    file_name="liftos_chat_history.txt",
                    mime="text/plain"
                )
            else:
                st.info("No chat history to export")
        
        # Chat statistics
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📊 Chat Stats")
            
            user_messages = len([msg for msg in st.session_state.chat_history if msg['role'] == 'user'])
            assistant_messages = len([msg for msg in st.session_state.chat_history if msg['role'] == 'assistant'])
            
            st.metric("Questions Asked", user_messages)
            st.metric("Responses Given", assistant_messages)

if __name__ == "__main__":
    main()