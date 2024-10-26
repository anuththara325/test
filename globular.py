import streamlit as st
import joblib
import numpy as np


def globular_cluster_analysis():
    model = joblib.load("stacking_clf_3.pkl")
    st.markdown(
        """
        <h1 style="text-align: center;">Globular Star Cluster Classifier</h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="text-align: center; font-weight: bold; font-size: 18px;">
            Apply clustering and classification algorithms and photometric analysis to detect and categorize globular clusters in galaxies. Use advanced machine learning techniques to assess their properties to provide deeper insights. <br>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create four columns for the description
    col1, col2, col3 = st.columns(3)

    with col1:

        # Input fields for the user
        CI_g = st.text_input(
            "Concentration Index CI_g",
            placeholder="Measure of light concentration in the g-band (green filter).",
            help="CI_g (Concentration Index in the g-band): This is a general measure of how much light is concentrated in a globular cluster, based on the bluish-green light (g-band). It gives an overall sense of how tightly packed or spread out the light is.",
        )
        CI_z = st.text_input(
            "Concentration Index CI_z",
            placeholder="Measure of light concentration in the z-band (infrared filter).",
            help="CI_z (Concentration Index in the z-band): Similar to CI_g, but using reddish light (z-band). It provides a general view of how concentrated the light is in the cluster when looking at it through the z-band filter, showing how light behaves across different wavelengths.",
        )
        m3_g = st.text_input(
            "Magnitude m3_g",
            placeholder="Brightness in the g-band for the third magnitude point.",
            help="m3_g (Magnitude in the g-band, 3-pixel aperture): This measures how bright a globular cluster appears when looking through a 3-pixel-wide circle, using greenish light (g-band). It gives a sense of the cluster's brightness in this specific part of the light spectrum.",
        )
        m3_z = st.text_input(
            "Magnitude m3_z",
            placeholder="Brightness in the z-band for the third magnitude point.",
            help="m3_z (Magnitude in the z-band, 3-pixel aperture): This is similar to m3_g, but it measures the brightness using redder light (z-band) through a 3-pixel-wide circle. It tells us how bright the cluster is in this part of the light spectrum.",
        )

    with col2:

        CI4_g = st.text_input(
            "Concentration Index CI4_g",
            placeholder="Light concentration in the g-band for the fourth index point.",
            help="CI4_g (Concentration Index in the g-band): This parameter compares the brightness of a globular cluster between a 4-pixel circle and a smaller 1-pixel circle, using bluish-green light (g-band). It shows how much light is concentrated in a slightly larger area, giving insight into the cluster's structure.",
        )
        CI4_z = st.text_input(
            "Concentration Index CI4_z",
            placeholder="Light concentration in the z-band for the fourth index point.",
            help="CI4_z (Concentration Index in the z-band): This parameter does the same as CI4_g, but it uses reddish light (z-band) instead. It compares the brightness in a 4-pixel circle versus a smaller 1-pixel circle, helping us see how the cluster's light behaves in a different color range.",
        )
        m4_g = st.text_input(
            "Magnitude m4_g",
            placeholder="Brightness in the g-band for the fourth magnitude point.",
            help="m4_g (Magnitude in the g-band, 4-pixel aperture): This measures the brightness of the cluster using a slightly larger 4-pixel-wide circle in the g-band (greenish light). It helps to see how the cluster's brightness changes with a larger area.",
        )
        m4_z = st.text_input(
            "Magnitude m4_z",
            placeholder="Brightness in the z-band for the fourth magnitude point.",
            help="m4_z (Magnitude in the z-band, 4-pixel aperture): This is like m4_g but uses a 4-pixel-wide circle and redder light (z-band) to measure the cluster's brightness, giving more insight into how the light behaves across different wavelengths.",
        )

    with col3:

        CI5_g = st.text_input(
            "Concentration Index CI5_g",
            placeholder="Light concentration in the g-band for the fifth index point.",
        )
        CI5_z = st.text_input(
            "Concentration Index CI5_z",
            placeholder="Light concentration in the z-band for the fifth index point.",
        )
        m5_g = st.text_input(
            "Magnitude m5_g",
            placeholder="Brightness in the g-band for the fifth magnitude point.",
        )
        m5_z = st.text_input(
            "Magnitude m5_z",
            placeholder="Brightness in the z-band for the fifth magnitude point.",
        )

    with col2:

        col_1, col_2, col_3 = st.columns(3)

        with col_2:

            # Button to trigger prediction
            if st.button("Classify GC"):
                try:
                    # Convert inputs to floats and make them into a feature list
                    input_features = [
                        float(CI_g),
                        float(CI_z),
                        float(m3_g),
                        float(m3_z),
                        float(CI4_g),
                        float(CI4_z),
                        float(m4_g),
                        float(m4_z),
                        float(CI5_g),
                        float(CI5_z),
                        float(m5_g),
                        float(m5_z),
                    ]

                    # Since the scaler is part of the model pipeline, we can directly predict
                    prediction = model.predict([input_features])

                    # Convert prediction to human-readable format (Red/Blue)
                    result = "Blue" if prediction[0] == 0 else "Red"

                    # Display the result
                    st.success(f"The GC is classified as {result}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
