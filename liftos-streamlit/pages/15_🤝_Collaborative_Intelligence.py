import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import sys
import os
import json
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_client import LiftOSAPIClient

# Page configuration
st.set_page_config(
    page_title="LiftOS - Collaborative Intelligence",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .collaboration-header {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .intelligence-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 4px solid #00cec9;
    }
    
    .collaboration-panel {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #a29bfe;
        margin: 1rem 0;
    }
    
    .ai-suggestion {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .human-input {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .consensus-meter {
        background: #e9ecef;
        border-radius: 10px;
        height: 25px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .consensus-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    .consensus-high { background: linear-gradient(90deg, #00b894, #00a085); }
    .consensus-medium { background: linear-gradient(90deg, #74b9ff, #0984e3); }
    .consensus-low { background: linear-gradient(90deg, #fd79a8, #e84393); }
    
    .decision-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 0.25rem;
        display: inline-block;
    }
    .status-approved { background: #d4edda; color: #155724; }
    .status-pending { background: #fff3cd; color: #856404; }
    .status-rejected { background: #f8d7da; color: #721c24; }
    
    .intelligence-flow {
        background: linear-gradient(135deg, #00cec9 0%, #00b894 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .policy-banner {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .team-member {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .team-member.online {
        border-color: #00b894;
    }
    
    .team-member.offline {
        border-color: #ddd;
        opacity: 0.7;
    }
    
    .chat-message {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 3px solid #74b9ff;
    }
    
    .chat-message.ai {
        border-left-color: #a29bfe;
        background: #f3f2ff;
    }
    
    .chat-message.human {
        border-left-color: #00b894;
        background: #f0fdf4;
    }
    
    .decision-tree {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .confidence-indicator {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .confidence-high { background: #d4edda; color: #155724; }
    .confidence-medium { background: #fff3cd; color: #856404; }
    .confidence-low { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

def get_team_insights(api_client, org_id="default_org"):
    """Get collaborative team insights"""
    try:
        response = api_client.get(f"/intelligence/collaboration/insights/{org_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching team insights: {str(e)}")
        return None

def get_ai_recommendations(api_client, org_id="default_org"):
    """Get AI recommendations for collaborative review"""
    try:
        response = api_client.get(f"/intelligence/collaboration/recommendations/{org_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching AI recommendations: {str(e)}")
        return None

def submit_human_feedback(api_client, feedback_data, org_id="default_org"):
    """Submit human feedback to AI system"""
    try:
        response = api_client.post(f"/intelligence/collaboration/feedback/{org_id}", 
                                 json=feedback_data)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        return None

def get_decision_consensus(api_client, decision_id, org_id="default_org"):
    """Get consensus data for a decision"""
    try:
        response = api_client.get(f"/intelligence/collaboration/consensus/{decision_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching consensus: {str(e)}")
        return None

def render_team_collaboration_overview():
    """Render team collaboration overview"""
    st.subheader("👥 Team Collaboration Overview")
    
    # Mock team data
    team_members = [
        {
            'name': 'Sarah Chen',
            'role': 'Marketing Director',
            'status': 'online',
            'last_active': datetime.now() - timedelta(minutes=5),
            'contributions': 23,
            'ai_alignment': 0.87
        },
        {
            'name': 'Marcus Rodriguez',
            'role': 'Performance Manager',
            'status': 'online',
            'last_active': datetime.now() - timedelta(minutes=12),
            'contributions': 31,
            'ai_alignment': 0.92
        },
        {
            'name': 'Emily Watson',
            'role': 'Data Analyst',
            'status': 'offline',
            'last_active': datetime.now() - timedelta(hours=2),
            'contributions': 18,
            'ai_alignment': 0.78
        },
        {
            'name': 'David Kim',
            'role': 'Creative Director',
            'status': 'online',
            'last_active': datetime.now() - timedelta(minutes=8),
            'contributions': 15,
            'ai_alignment': 0.65
        }
    ]
    
    # Team stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        online_members = len([m for m in team_members if m['status'] == 'online'])
        st.metric("Online Members", f"{online_members}/{len(team_members)}")
    
    with col2:
        total_contributions = sum(m['contributions'] for m in team_members)
        st.metric("Total Contributions", total_contributions)
    
    with col3:
        avg_alignment = np.mean([m['ai_alignment'] for m in team_members])
        st.metric("AI Alignment", f"{avg_alignment:.0%}")
    
    with col4:
        active_decisions = 7  # Mock
        st.metric("Active Decisions", active_decisions)
    
    # Team members grid
    st.markdown("### 👥 Team Members")
    
    cols = st.columns(2)
    for i, member in enumerate(team_members):
        with cols[i % 2]:
            status_class = 'online' if member['status'] == 'online' else 'offline'
            status_emoji = '🟢' if member['status'] == 'online' else '⚫'
            
            st.markdown(f"""
            <div class="team-member {status_class}">
                <h4>{status_emoji} {member['name']}</h4>
                <p><strong>Role:</strong> {member['role']}</p>
                <p><strong>Contributions:</strong> {member['contributions']}</p>
                <p><strong>AI Alignment:</strong> {member['ai_alignment']:.0%}</p>
                <p><strong>Last Active:</strong> {member['last_active'].strftime('%H:%M')}</p>
            </div>
            """, unsafe_allow_html=True)

def render_ai_human_collaboration():
    """Render AI-Human collaboration interface"""
    st.subheader("🤖🤝👤 AI-Human Collaboration")
    
    # Mock AI recommendations
    ai_recommendations = [
        {
            'id': 'ai_rec_001',
            'title': 'Increase Google Ads Budget by 15%',
            'description': 'AI analysis suggests increasing Google Ads budget based on performance trends',
            'confidence': 0.89,
            'reasoning': [
                'Google Ads ROAS increased 23% over last 7 days',
                'Search volume trending up 18% for target keywords',
                'Competitor analysis shows opportunity gap'
            ],
            'human_feedback': [],
            'status': 'pending_review',
            'ai_certainty': 'high'
        },
        {
            'id': 'ai_rec_002',
            'title': 'Pause Underperforming Creative Set',
            'description': 'Creative fatigue detected in Meta campaign creative set #47',
            'confidence': 0.76,
            'reasoning': [
                'CTR declined 34% over 5 days',
                'Frequency reached 4.2x (above optimal)',
                'Similar creatives showing better performance'
            ],
            'human_feedback': [
                {
                    'user': 'Sarah Chen',
                    'feedback': 'Agree, but let\'s test one more variation first',
                    'timestamp': datetime.now() - timedelta(minutes=15)
                }
            ],
            'status': 'under_discussion',
            'ai_certainty': 'medium'
        },
        {
            'id': 'ai_rec_003',
            'title': 'Launch Retargeting Campaign',
            'description': 'High-value audience segment identified for retargeting',
            'confidence': 0.94,
            'reasoning': [
                'Segment shows 3.2x higher LTV',
                'Optimal retargeting window identified',
                'Budget availability confirmed'
            ],
            'human_feedback': [
                {
                    'user': 'Marcus Rodriguez',
                    'feedback': 'Approved - excellent analysis',
                    'timestamp': datetime.now() - timedelta(minutes=8)
                },
                {
                    'user': 'Emily Watson',
                    'feedback': 'Data looks solid, proceed',
                    'timestamp': datetime.now() - timedelta(minutes=12)
                }
            ],
            'status': 'approved',
            'ai_certainty': 'high'
        }
    ]
    
    for rec in ai_recommendations:
        status_class = 'status-approved' if rec['status'] == 'approved' else 'status-pending' if rec['status'] == 'pending_review' else 'status-pending'
        confidence_class = 'confidence-high' if rec['confidence'] > 0.8 else 'confidence-medium' if rec['confidence'] > 0.6 else 'confidence-low'
        
        with st.expander(f"🤖 {rec['title']} (Confidence: {rec['confidence']:.0%})", expanded=(rec['status'] == 'pending_review')):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="ai-suggestion">
                    <h4>{rec['description']}</h4>
                    <p><strong>Status:</strong> <span class="decision-status {status_class}">{rec['status'].replace('_', ' ').upper()}</span></p>
                    <p><strong>AI Certainty:</strong> <span class="confidence-indicator {confidence_class}">{rec['ai_certainty'].upper()}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # AI reasoning
                st.markdown("**AI Reasoning:**")
                for reason in rec['reasoning']:
                    st.markdown(f"• {reason}")
                
                # Human feedback section
                if rec['human_feedback']:
                    st.markdown("**Team Feedback:**")
                    for feedback in rec['human_feedback']:
                        st.markdown(f"""
                        <div class="chat-message human">
                            <strong>{feedback['user']}:</strong> {feedback['feedback']}
                            <br><small>{feedback['timestamp'].strftime('%H:%M')}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                # Confidence meter
                confidence = rec['confidence']
                confidence_class_meter = 'consensus-high' if confidence > 0.8 else 'consensus-medium' if confidence > 0.6 else 'consensus-low'
                
                st.markdown(f"""
                <div class="consensus-meter">
                    <div class="consensus-fill {confidence_class_meter}" style="width: {confidence*100}%">
                        {confidence:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                if rec['status'] == 'pending_review':
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"✅ Approve", key=f"approve_{rec['id']}"):
                            approve_ai_recommendation(rec)
                    
                    with col_b:
                        if st.button(f"❌ Reject", key=f"reject_{rec['id']}"):
                            reject_ai_recommendation(rec)
                    
                    # Feedback input
                    feedback_text = st.text_area(
                        "Add Feedback:",
                        key=f"feedback_{rec['id']}",
                        placeholder="Share your thoughts..."
                    )
                    
                    if st.button(f"💬 Submit Feedback", key=f"submit_{rec['id']}"):
                        if feedback_text:
                            submit_feedback_to_ai(rec, feedback_text)

def approve_ai_recommendation(recommendation):
    """Approve AI recommendation"""
    st.success(f"✅ Approved: {recommendation['title']}")
    st.info("🤖 AI will execute this recommendation and monitor results")

def reject_ai_recommendation(recommendation):
    """Reject AI recommendation"""
    st.warning(f"❌ Rejected: {recommendation['title']}")
    st.info("🤖 AI will learn from this feedback and adjust future recommendations")

def submit_feedback_to_ai(recommendation, feedback_text):
    """Submit feedback to AI system"""
    st.success("💬 Feedback submitted to AI system")
    st.info("🤖 AI will incorporate this feedback into future analysis")

def render_collaborative_decision_making():
    """Render collaborative decision making interface"""
    st.subheader("🎯 Collaborative Decision Making")
    
    # Mock active decisions
    active_decisions = [
        {
            'id': 'decision_001',
            'title': 'Q4 Budget Allocation Strategy',
            'description': 'Determine optimal budget allocation for Q4 campaigns',
            'participants': ['Sarah Chen', 'Marcus Rodriguez', 'Emily Watson'],
            'ai_input': 'Recommends 40% Google, 35% Meta, 25% Other based on historical performance',
            'human_votes': {
                'Sarah Chen': {'google': 0.45, 'meta': 0.30, 'other': 0.25},
                'Marcus Rodriguez': {'google': 0.42, 'meta': 0.38, 'other': 0.20},
                'Emily Watson': {'google': 0.40, 'meta': 0.35, 'other': 0.25}
            },
            'consensus_level': 0.78,
            'deadline': datetime.now() + timedelta(days=2),
            'status': 'in_progress'
        },
        {
            'id': 'decision_002',
            'title': 'New Creative Testing Framework',
            'description': 'Establish framework for testing new creative variations',
            'participants': ['David Kim', 'Sarah Chen', 'Marcus Rodriguez'],
            'ai_input': 'Suggests A/B testing with 70/30 split and 7-day test periods',
            'human_votes': {
                'David Kim': 'approved',
                'Sarah Chen': 'approved',
                'Marcus Rodriguez': 'pending'
            },
            'consensus_level': 0.67,
            'deadline': datetime.now() + timedelta(days=5),
            'status': 'pending_consensus'
        }
    ]
    
    for decision in active_decisions:
        consensus_class = 'consensus-high' if decision['consensus_level'] > 0.8 else 'consensus-medium' if decision['consensus_level'] > 0.6 else 'consensus-low'
        
        with st.expander(f"🎯 {decision['title']} (Consensus: {decision['consensus_level']:.0%})", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="decision-tree">
                    <h4>{decision['description']}</h4>
                    <p><strong>Participants:</strong> {', '.join(decision['participants'])}</p>
                    <p><strong>Deadline:</strong> {decision['deadline'].strftime('%Y-%m-%d %H:%M')}</p>
                    <p><strong>Status:</strong> {decision['status'].replace('_', ' ').title()}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # AI input
                st.markdown(f"""
                <div class="chat-message ai">
                    <strong>🤖 AI Input:</strong> {decision['ai_input']}
                </div>
                """, unsafe_allow_html=True)
                
                # Human votes/input
                if 'human_votes' in decision:
                    st.markdown("**Team Input:**")
                    for participant, vote in decision['human_votes'].items():
                        if isinstance(vote, dict):
                            vote_str = ', '.join([f"{k}: {v:.0%}" for k, v in vote.items()])
                        else:
                            vote_str = vote
                        
                        st.markdown(f"""
                        <div class="chat-message human">
                            <strong>{participant}:</strong> {vote_str}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                # Consensus meter
                st.markdown(f"""
                <div class="consensus-meter">
                    <div class="consensus-fill {consensus_class}" style="width: {decision['consensus_level']*100}%">
                        {decision['consensus_level']:.0%}
                    </div>
                </div>
                <p><strong>Consensus Level</strong></p>
                """, unsafe_allow_html=True)
                
                # Time remaining
                time_remaining = decision['deadline'] - datetime.now()
                st.markdown(f"**Time Remaining:** {time_remaining.days}d {time_remaining.seconds//3600}h")
                
                # Action buttons
                if decision['status'] == 'in_progress':
                    if st.button(f"🗳️ Cast Vote", key=f"vote_{decision['id']}"):
                        cast_vote_interface(decision)
                
                if decision['consensus_level'] > 0.75:
                    if st.button(f"✅ Finalize Decision", key=f"finalize_{decision['id']}"):
                        finalize_decision(decision)

def cast_vote_interface(decision):
    """Interface for casting votes on decisions"""
    st.markdown("### 🗳️ Cast Your Vote")
    
    with st.form(f"vote_form_{decision['id']}"):
        st.markdown(f"**Decision:** {decision['title']}")
        
        if 'budget' in decision['title'].lower():
            st.markdown("**Budget Allocation Preferences:**")
            google_pct = st.slider("Google Ads %", 0, 100, 40)
            meta_pct = st.slider("Meta Ads %", 0, 100, 35)
            other_pct = st.slider("Other Platforms %", 0, 100, 25)
            
            total_pct = google_pct + meta_pct + other_pct
            if total_pct != 100:
                st.warning(f"Total allocation: {total_pct}% (should be 100%)")
        else:
            vote_choice = st.radio(
                "Your Decision:",
                ["Approve", "Reject", "Needs Modification"]
            )
        
        comments = st.text_area("Comments (optional):")
        
        if st.form_submit_button("Submit Vote"):
            st.success("✅ Vote submitted successfully!")
            st.info("🤖 AI will incorporate your input into the consensus calculation")

def finalize_decision(decision):
    """Finalize a collaborative decision"""
    st.success(f"✅ Decision finalized: {decision['title']}")
    st.info("🤖 AI will implement the agreed-upon strategy and monitor results")

def render_intelligence_synthesis():
    """Render intelligence synthesis dashboard"""
    st.subheader("🧠 Intelligence Synthesis")
    
    # Mock synthesis data
    synthesis_data = {
        'ai_insights': 47,
        'human_insights': 23,
        'combined_insights': 12,
        'accuracy_improvement': 0.34,
        'decision_speed': 0.67,
        'consensus_rate': 0.82
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "AI Insights Generated",
            synthesis_data['ai_insights'],
            delta="+12 this week"
        )
        
        st.metric(
            "Human Insights Added",
            synthesis_data['human_insights'],
            delta="+5 this week"
        )
    
    with col2:
        st.metric(
            "Combined Insights",
            synthesis_data['combined_insights'],
            delta="+3 this week"
        )
        
        st.metric(
            "Accuracy Improvement",
            f"+{synthesis_data['accuracy_improvement']:.0%}",
            help="Improvement from AI-Human collaboration"
        )
    
    with col3:
        st.metric(
            "Decision Speed",
            f"{synthesis_data['decision_speed']:.0%}",
            delta="+15% faster",
            help="Speed improvement from collaboration"
        )
        
        st.metric(
            "Consensus Rate",
            f"{synthesis_data['consensus_rate']:.0%}",
            help="Rate of reaching team consensus"
        )
    
    # Intelligence synthesis chart
    st.markdown("### 📊 Intelligence Synthesis Over Time")
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    ai_insights = np.random.poisson(2, len(dates))
    human_insights = np.random.poisson(1, len(dates))
    combined_insights = np.random.poisson(0.5, len(dates))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=ai_insights.cumsum(),
        mode='lines+markers',
        name='AI Insights',
        line=dict(color='#a29bfe', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=human_insights.cumsum(),
        mode='lines+markers',
        name='Human Insights',
        line=dict(color='#00b894', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=combined_insights.cumsum(),
        mode='lines+markers',
        name='Combined Insights',
        line=dict(color='#fd79a8', width=3)
    ))
    
    fig.update_layout(
        title="Cumulative Intelligence Generation",
        xaxis_title="Date",
        yaxis_title="Insights Generated",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_real_time_collaboration():
    """Render real-time collaboration feed"""
    st.subheader("💬 Real-time Collaboration Feed")
    
    # Mock collaboration events
    events = [
        {
            'timestamp': datetime.now() - timedelta(minutes=2),
            'type': 'ai_insight',
            'user': 'AI System',
            'message': 'Detected 15% increase in mobile traffic - suggest mobile-first creative testing',
            'confidence': 0.87
        },
        {
            'timestamp': datetime.now() - timedelta(minutes=5),
            'type': 'human_feedback',
            'user': 'Sarah Chen',
            'message': 'Agreed on the Google Ads budget increase - let\'s monitor closely',
            'confidence': None
        },
        {
            'timestamp': datetime.now() - timedelta(minutes=8),
            'type': 'decision',
            'user': 'System',
            'message': 'Decision "Q4 Budget Allocation" reached 78% consensus',
            'confidence': None
        },
        {
            'timestamp': datetime.now() - timedelta(minutes=12),
            'type': 'ai_insight',
            'user': 'AI System',
            'message': 'Klaviyo engagement rates up 23% - optimal time for email campaign launch',
            'confidence': 0.92
        },
        {
            'timestamp': datetime.now() - timedelta(minutes=15),
            'type': 'human_feedback',
            'user': 'Marcus Rodriguez',
            'message': 'Creative fatigue confirmed in Meta campaigns - pausing as recommended',
            'confidence': None
        }
    ]
    
    for event in events:
        message_class = 'ai' if event['type'] == 'ai_insight' else 'human' if event['type'] == 'human_feedback' else ''
        
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <strong>{event['user']}</strong> • {event['timestamp'].strftime('%H:%M')}
            <br>{event['message']}
            {f'<br><span class="confidence-indicator confidence-high">Confidence: {event["confidence"]:.0%}</span>' if event['confidence'] else ''}
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="collaboration-header">
        <h1>🤝 Collaborative Intelligence Interfaces</h1>
        <p>AI-Human collaboration for enhanced decision making and strategic optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Policy banner
    st.markdown("""
    <div class="policy-banner">
        <strong>🎯 Policy 2 & 5 Integration:</strong> Democratize Speed + Memory-Driven Intelligence - 
        Collaborative intelligence with 0.041s response and human-AI synthesis
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize API client
    api_client = LiftOSAPIClient()
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        org_id = st.selectbox(
            "Organization:",
            ["default_org", "demo_org", "test_org"],
            help="Select organization for collaboration"
        )
    
    with col2:
        collaboration_mode = st.selectbox(
            "Collaboration Mode:",
            ["Real-time", "Asynchronous", "Hybrid"],
            index=0,
            help="Collaboration timing mode"
        )
    
    with col3:
        ai_assistance_level = st.selectbox(
            "AI Assistance:",
            ["High", "Medium", "Low"],
            index=1,
            help="Level of AI assistance in decisions"
        )
    
    with col4:
        if st.button("🔄 Refresh Feed", type="primary"):
            st.rerun()
    
    # Get data
    with st.spinner("Loading collaboration data..."):
        team_insights = get_team_insights(api_client, org_id)
        ai_recommendations = get_ai_recommendations(api_client, org_id)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 Team Overview",
        "🤖🤝👤 AI-Human Collaboration",
        "🎯 Decision Making",
        "🧠 Intelligence Synthesis",
        "💬 Real-time Feed"
    ])
    
    with tab1:
        render_team_collaboration_overview()
    
    with tab2:
        render_ai_human_collaboration()
    
    with tab3:
        render_collaborative_decision_making()
    
    with tab4:
        render_intelligence_synthesis()
    
    with tab5:
        render_real_time_collaboration()

if __name__ == "__main__":
    main()