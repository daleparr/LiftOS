"""
LiftOS Bayesian Analysis Interface

Comprehensive Bayesian prior analysis, conflict detection, and SBC validation.
Supports both free audit tier for prospects and advanced analysis for clients.
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import uuid
from typing import Dict, List, Any, Optional

# Configuration
BAYESIAN_SERVICE_URL = "http://localhost:8010"
CAUSAL_SERVICE_URL = "http://localhost:8008"

# Helper functions
def call_bayesian_api(endpoint: str, data: Dict = None, method: str = "GET") -> Dict:
    """Call Bayesian Analysis Service API"""
    try:
        url = f"{BAYESIAN_SERVICE_URL}{endpoint}"
        if method == "POST":
            response = requests.post(url, json=data, timeout=5)
        else:
            response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": f"Connection Error: {str(e)}"}

# Main app
def main():
    # Header
    st.title("Bayesian Analysis Platform")
    st.markdown("### Advanced Bayesian prior analysis, conflict detection, and model validation")
    
    # Sidebar navigation
    st.sidebar.title("Bayesian Analysis")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Free Audit", "Advanced Analysis", "SBC Validation", "Evidence Assessment"]
    )
    
    # Service status check
    with st.sidebar:
        st.subheader("Service Status")
        if st.button("Check Service Status"):
            health_response = call_bayesian_api("/health")
            if health_response["success"]:
                st.success("Bayesian Service Online")
                service_info = health_response["data"]
                st.json(service_info.get("components", {}))
            else:
                st.error("Service Unavailable")
                st.write(health_response.get("error", "Unknown error"))
        else:
            st.info("Click to check service status")
    
    # Main content based on selection
    if analysis_type == "Free Audit":
        render_free_audit()
    elif analysis_type == "Advanced Analysis":
        render_advanced_analysis()
    elif analysis_type == "SBC Validation":
        render_sbc_validation()
    elif analysis_type == "Evidence Assessment":
        render_evidence_assessment()

def render_free_audit():
    """Render free audit interface for prospects"""
    st.subheader("Free Bayesian Prior Audit")
    st.write("Get instant insights into your marketing attribution beliefs vs. data evidence")
    st.info("No commitment required - 10-15 minutes - Immediate results")
    
    # Contact information
    st.subheader("Contact Information")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name *", placeholder="John Smith")
        company = st.text_input("Company *", placeholder="Example Corp")
        email = st.text_input("Email *", placeholder="john@example.com")
    
    with col2:
        phone = st.text_input("Phone", placeholder="+1 (555) 123-4567")
        title = st.text_input("Job Title", placeholder="Marketing Director")
        industry = st.selectbox("Industry", [
            "E-commerce", "Retail", "Technology", "Financial Services",
            "Healthcare", "Media & Entertainment", "Other"
        ])
    
    # Business context
    st.subheader("Business Context")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        annual_revenue = st.selectbox("Annual Revenue", [
            "< $1M", "$1M - $10M", "$10M - $100M", "$100M - $1B", "> $1B"
        ])
    
    with col2:
        marketing_budget = st.selectbox("Marketing Budget", [
            "< $100K", "$100K - $1M", "$1M - $10M", "$10M - $100M", "> $100M"
        ])
    
    with col3:
        channels = st.multiselect("Primary Channels", [
            "TV", "Digital Display", "Search", "Social Media", 
            "Email", "Direct Mail", "Radio", "Out-of-Home"
        ])
    
    # Prior elicitation (limited to 5 for free tier)
    st.subheader("Your Marketing Attribution Beliefs")
    st.info("Free audit includes up to 5 parameters. Upgrade for unlimited analysis.")
    
    parameters = []
    num_params = st.number_input("Number of parameters to analyze", 1, 5, 3)
    
    for i in range(num_params):
        st.write(f"**Parameter {i+1}**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            param_name = st.text_input(f"Parameter Name", 
                                     value=["TV Attribution", "Search Attribution", "Social Attribution"][i] if i < 3 else "",
                                     key=f"param_name_{i}")
        
        with col2:
            current_belief = st.slider(f"Your Current Belief (%)", 0, 100, 
                                     [40, 25, 15][i] if i < 3 else 20,
                                     key=f"belief_{i}")
        
        with col3:
            confidence = st.slider(f"Confidence Level", 0.0, 1.0, 0.7, 0.1,
                                 key=f"confidence_{i}")
        
        if param_name:
            parameters.append({
                "name": param_name,
                "current_belief": current_belief / 100,
                "confidence": confidence,
                "range_min": max(0, current_belief - 20) / 100,
                "range_max": min(100, current_belief + 20) / 100
            })
    
    # Run free audit
    if st.button("Start Free Audit", type="primary", use_container_width=True):
        if not all([name, company, email]) or not parameters:
            st.error("Please fill in all required fields and at least one parameter.")
            return
        
        # Create audit request
        audit_request = {
            "prospect_id": str(uuid.uuid4()),
            "contact_info": {
                "name": name,
                "company": company,
                "email": email,
                "phone": phone,
                "title": title
            },
            "business_context": {
                "industry": industry,
                "annual_revenue": annual_revenue,
                "marketing_budget": marketing_budget,
                "primary_channels": channels
            },
            "parameters": parameters,
            "source": "streamlit_frontend"
        }
        
        # Call prior elicitation API
        with st.spinner("Analyzing your priors..."):
            elicitation_response = call_bayesian_api("/api/v1/audit/elicit-priors", audit_request, "POST")
        
        if elicitation_response["success"]:
            st.success("Prior Elicitation Complete!")
            st.json(elicitation_response["data"])
        else:
            st.error(f"Audit failed: {elicitation_response.get('error')}")

def render_advanced_analysis():
    """Render advanced analysis interface for paid users"""
    st.subheader("Advanced Bayesian Analysis")
    st.write("Comprehensive prior updating, evidence assessment, and model validation")
    st.info("This section requires a paid subscription. Contact sales for access.")

def render_sbc_validation():
    """Render SBC validation interface"""
    st.subheader("Simulation Based Calibration (SBC)")
    st.write("Validate your Bayesian models for reliability and calibration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_simulations = st.slider("Number of Simulations", 100, 10000, 1000, 100)
        validation_level = st.selectbox("Validation Level", [
            "Basic", "Standard", "Comprehensive", "Research Grade"
        ])
    
    with col2:
        st.checkbox("Include Diagnostic Plots")
        st.checkbox("Parallel Execution")
    
    if st.button("Run SBC Validation", type="primary"):
        st.success("SBC Validation Complete!")
        st.write("Validation results would appear here...")

def render_evidence_assessment():
    """Render evidence assessment interface"""
    st.subheader("Evidence Strength Assessment")
    st.write("Analyze the quality and reliability of your evidence")
    
    evidence_types = st.multiselect("Evidence Types", [
        "Observational Data", "Experimental Results", "Historical Analysis",
        "Expert Opinion", "Literature Review", "Market Research"
    ])
    
    if st.button("Assess Evidence Strength", type="primary"):
        st.success("Evidence Assessment Complete!")
        st.write("Assessment results would appear here...")

if __name__ == "__main__":
    main()