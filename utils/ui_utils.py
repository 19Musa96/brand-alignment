import streamlit as st

def render_identity_profile(name,identity,expander_slot):
    
    with expander_slot.container():
        with st.expander(f"View identity profile"):
            st.markdown("---")
            st.markdown("#### Identity Profile")
            st.markdown(f"""
                #### 🌐 Primary Domain
                {identity['primary_domain']}

                #### 🧭 Sub Domains
                {", ".join(identity['sub_domains'])}

                #### 💰 Economic Model
                {identity['economic_model']}

                #### 🎯 Core Mission
                _{identity['core_mission']}_

                #### 🧠 Value Signals
                {", ".join(identity['value_signals'])}

                #### 🎭 Cultural Positioning
                {identity['cultural_positioning']}

                #### 🏛 Power Positioning
                **{identity['power_positioning']}**

                #### ⚠️ Controversy Themes
                {", ".join(identity['controversy_themes'])}
            """)