import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from langchain.chat_models import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

TAVILY_API_KEY = "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"
AZURE_OPENAI_API_KEY = "ad31085a728c400cb51886b9906951e4"
AZURE_ENDPOINT = "https://chatgpt-key.openai.azure.com/"
AZURE_DEPLOYMENT_NAME = "gpt-4o-2"
AZURE_API_VERSION = "2024-12-01-preview"

CONSULTING_DOMAINS = [
    "mckinsey.com", "pwc.com", "bcg.com", "kpmg.com", "ey.com",
    "forbes.com", "gartner.com", "deloitte.com", "hbr.org", "statista.com"
]


def tavily_market_research_tool(query: str) -> dict:
    url = "https://api.tavily.com/search"
    response = requests.post(
        url,
        json={"query": query, "search_depth": "advanced"},
        headers={"Authorization": f"Bearer {TAVILY_API_KEY}"}
    )
    return response.json()

def get_azure_llm(max_tokens=800, temperature=0.6):
    return AzureChatOpenAI(
        deployment_name=AZURE_DEPLOYMENT_NAME,
        api_version=AZURE_API_VERSION,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        model_name="gpt-4o",
        temperature=temperature,
        max_tokens=max_tokens
    )


def consumer_review_summary(query: str):
    if not query:
        st.warning("Please enter a Company Name or Company's Product to analyze the reviews.")
        return

    results = tavily_market_research_tool(query)
    top_results = results["results"][:5]

    source_text = "\n\n".join([
        f"- **Title**: {r['title']}\n  **URL**: {r['url']}\n  **Extract**: {r['content'][:300]}..."
        for r in top_results
    ])

    prompt = PromptTemplate.from_template("""
    You are a product review analyst. Based on the extracted data below, write a structured and neutral summary, including:
    1. General Overview
    2. Key Strengths
    3. Potential Weaknesses
    4. Conclusion
    Avoid quoting user reviews directly. Focus on summarizing the overall opinion. Mention the URL.

    Extracted content:
    {source_text}
    """)

    chain = prompt | get_azure_llm() | StrOutputParser()
    summary = chain.invoke({"source_text": source_text})

    st.subheader("📝 Competitive Intelligence Result")
    st.markdown(summary)

def strategy_consulting_output(query: str):
    if not query:
        st.warning("Please enter a Company Name for SWOT Analysis and Market Positioning'.")
        return

    results = tavily_market_research_tool(query)
    filtered = [
        r for r in results["results"]
        if any(domain in r["url"] for domain in CONSULTING_DOMAINS)
    ]
    top_results = filtered[:5] if filtered else results["results"][:5]

    source_text = "\n\n".join([
        f"- **Title**: {r['title']}\n  **URL**: {r['url']}\n  **Extract**: {r['content'][:300]}..."
        for r in top_results
    ])

    prompt = PromptTemplate.from_template("""
    You are a senior strategy consultant at McKinsey. Write a business-style strategic summary based on the content below:

    1. Company Overview
    2. Market Positioning
    3. SWOT Analysis
    4. Competitive Landscape
    5. Recommendations

    Don't quote directly. Mention URLs.

    Extracted content:
    {source_text}
    """)

    chain = prompt | get_azure_llm(max_tokens=1000) | StrOutputParser()
    summary = chain.invoke({"source_text": source_text})

    st.subheader("Strategic Consulting Report")
    st.markdown(summary)


def generate_chart_data(query):
    tavily_results = tavily_market_research_tool(query)
    filtered_results = [
        r for r in tavily_results["results"]
        if any(domain in r["url"] for domain in CONSULTING_DOMAINS)
    ]
    top_results = filtered_results[:5] if filtered_results else tavily_results["results"][:5]

    sources = [
        f"- **Title**: {r['title']}\n  **URL**: {r['url']}\n  **Extract**: {r['content'][:300]}..."
        for r in top_results
    ]
    joined_sources = "\n\n".join(sources)

    chart_prompt = PromptTemplate.from_template("""
    You are a data analyst tasked with extracting structured data from consulting research for visualization. Based on the text below, extract:

    1. Pie Chart: Revenue by product/segment/region
    2. Line Chart: Revenue/Stock trend over time
    3. BCG Matrix: Products with market share and growth
    4. Waterfall Chart: Revenue → cost/profit breakdown
    5. Radar Chart: SWOT or Competitor Comparison
    6. SWOT: Strengths, Weaknesses, Opportunities, Threats

    Output a JSON object like:
    {{
      "pie_chart": {{ "labels": [...], "values": [...] }},
      "line_chart": {{ "years": [...], "revenue": [...] }},
      "bcg_matrix": [{{ "name": "...", "market_share": ..., "market_growth": ... }}],
      "waterfall_chart": {{ "stages": [...], "values": [...] }},
      "radar_chart": {{
          "categories": [...],
          "company": [...],
          "competitor": [...]
      }},
      "swot": {{
        "Strengths": [...],
        "Weaknesses": [...],
        "Opportunities": [...],
        "Threats": [...]
      }}
    }}

    Respond with only JSON and NO explanation.

    Extracted content:
    {source_text}
    """)

    chain = chart_prompt | get_azure_llm() | StrOutputParser()
    result = chain.invoke({"source_text": joined_sources})

    if result.startswith("```json"):
        result = result.strip("`").strip("json").strip()
    if result.endswith("```"):
        result = result[:-3].strip()

    try:
        chart_data = json.loads(result)
        with open("chart_data.json", "w") as f:
            json.dump(chart_data, f, indent=4)
        return chart_data
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        return None
    

def render_visualizations(chart_data):
    st.subheader("📊 Slide Through Visual Insights")
    slider_tabs = ["Pie Chart", "Line Chart", "BCG Matrix", "Waterfall", "Radar", "SWOT"]
    selected_tab = st.select_slider("Choose Visualization", options=slider_tabs)

    if selected_tab == "Pie Chart" and "pie_chart" in chart_data:
        plt.figure(figsize=(5, 5))
        plt.pie(chart_data["pie_chart"]["values"],
                labels=chart_data["pie_chart"]["labels"], autopct='%1.1f%%')
        plt.title("Revenue by Category")
        st.pyplot(plt)

    elif selected_tab == "Line Chart" and "line_chart" in chart_data:
        years = chart_data["line_chart"]["years"]
        revenue = chart_data["line_chart"]["revenue"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=revenue, mode="lines+markers", name="Revenue"))
        fig.update_layout(title="📈 Revenue Over Time", xaxis_title="Year", yaxis_title="Revenue")
        st.plotly_chart(fig)

    elif selected_tab == "BCG Matrix" and "bcg_matrix" in chart_data:
        df = pd.DataFrame(chart_data["bcg_matrix"])
        fig = go.Figure(data=go.Scatter(
            x=df["market_share"], y=df["market_growth"],
            mode='markers+text', text=df["name"], marker=dict(size=20)
        ))
        fig.update_layout(title="🧮 BCG Matrix", xaxis_title="Market Share", yaxis_title="Market Growth")
        st.plotly_chart(fig)

    elif selected_tab == "Waterfall" and "waterfall_chart" in chart_data:
        wf = chart_data["waterfall_chart"]
        fig = go.Figure(go.Waterfall(
            name="💰 Profit Breakdown",
            orientation="v",
            measure=["relative"] * len(wf["values"]),
            x=wf["stages"],
            y=wf["values"],
            connector={"line": {"color": "gray"}}
        ))
        fig.update_layout(title="💸 Financial Waterfall", waterfallgap=0.3)
        st.plotly_chart(fig)

    elif selected_tab == "Radar" and "radar_chart" in chart_data:
        radar = chart_data["radar_chart"]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar["company"], theta=radar["categories"],
            fill='toself', name='Company'
        ))
        fig.add_trace(go.Scatterpolar(
            r=radar["competitor"], theta=radar["categories"],
            fill='toself', name='Competitor'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig)

    elif selected_tab == "SWOT" and "swot" in chart_data:
        st.markdown("### 🔍 SWOT Analysis")
        for key, values in chart_data["swot"].items():
            st.markdown(f"**{key}:**")
            for v in values:
                st.markdown(f"- {v}")
def main():
    st.set_page_config(page_title="AI Market Insights | Competitive Intelligence Bot", layout="wide")


    st.markdown("""
        <h1 style='text-align: center; color: #0056b3;'>🤖 Market & Competitive Intelligence Platform</h1>
        <p style='text-align: center; font-size: 18px;'>Get Strategic Competitve Intelligence, Consulting Insights,Graphical Insight </p>
    """, unsafe_allow_html=True)


    menu = st.sidebar.radio("📂 Navigate", ["Home", "Competitve Intelligence", "Consulting Insights","Graphical Insight"])


    if menu == "Home":
        st.markdown("### 👋 Welcome to the Future of Market Research")
        st.image("https://images.unsplash.com/photo-1603791440384-56cd371ee9a7", use_container_width=True)
        st.markdown("""
        This tool allows you to:
        - Summarize product reviews from blogs and platforms
        - Generate strategic business analysis reports from real industry sources
        - Use AI to enhance your research and decision-making
        """)
        st.success("Use the sidebar to get started with review or strategy analysis.")


    elif menu == "Competitve Intelligence":
        st.markdown("### 🛍️ Competitve Intelligence")
        query = st.text_input("Enter a Company Name or Company's Product:", key="review_input")
        if query:
            with st.spinner("Analyzing Competitve Intelligence..."):
                consumer_review_summary(query)


    elif menu == "Consulting Insights":
        st.markdown("### 📊 Strategy Consulting Analysis")
        query = st.text_input("Enter a company or trend to explore:", key="consulting_input")
        if query:
            with st.spinner("Generating consulting insight..."):
                strategy_consulting_output(query)


    elif menu == "Graphical Insight":
        st.markdown("### 🛍️ Graphical Interpretation")
        query = st.text_input("Enter a product/service to analyze:", key="review_input")
        if query:
            with st.spinner("Graphical Overview..."):
                generate_chart_data(query)
        try:
            with open("chart_data.json", "r") as f:
                chart_data = json.load(f)
            render_visualizations(chart_data)
        except Exception as e:
            st.error("⚠️ No chart_data.json found. Please analyze a topic first.")


if __name__ == "__main__":
    main()
