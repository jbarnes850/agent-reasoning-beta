"""
Homepage for the Agent Reasoning Beta platform.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Agent Reasoning Beta",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main content
st.title("🤖 Agent Reasoning Beta")
st.markdown("""
Welcome to the Agent Reasoning Beta platform! This platform provides tools for:

- 🎮 **Interactive Agent Experiments**: Run and visualize agent reasoning processes
- 📊 **Real-time Analytics**: Monitor performance and resource utilization
- ⚙️ **Flexible Configuration**: Customize agent behavior and system settings

### Getting Started

1. Visit the **Playground** to run agent experiments
2. Check the **Analytics** page for performance insights
3. Configure your settings in the **Settings** page

### Quick Links

- [Documentation](docs/README.md)
- [GitHub Repository](https://github.com/jbarnes850/agent-reasoning-beta)
- [Issue Tracker](https://github.com/jbarnes850/agent-reasoning-beta/issues)
""")

# System status
st.sidebar.header("System Status")
if st.session_state.get("experiment_running", False):
    st.sidebar.success("🟢 System Active")
    st.sidebar.metric(
        "Active Agents",
        len(st.session_state.get("agents", []))
    )
else:
    st.sidebar.info("⚪ System Idle")
