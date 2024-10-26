import streamlit as st
from classifier import galaxy_classifier
from enhancer import image_enhancer
from globular import globular_cluster_analysis
from cluster import galaxy_cluster_analysis
from chatbot import astro_chatbot
import base64


# Set page layout and title
st.set_page_config(
    page_title="GALACTIC-X",
    layout="wide",
    page_icon="gs-page-logo.png",
)
# # Configure Streamlit page settings
# st.set_page_config(
#     page_title="Chat with Gemini-Pro!",
#     page_icon=":brain:",  # Favicon emoji
#     layout="centered",  # Page layout option
# )


# Function to add background image (using base64 encoding)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Function to display the home page
def home_page():
    # st.title("Galactic Scholors 🌌")
    # Center-aligned title using HTML
    # one, two, three, four, five, six, seven = st.columns(7)
    # with four:
    #     st.image(
    #         "gs-logo.png",
    #         width=100,
    #     )
    # CSS for glowing effect
    glowing_title = """
    <style>
    .glow {
        font-size: 100px;
        font-weight: bold;
        text-align: center;
        color: white;
        text-shadow: 0 0 9px white, 0 0 20px #00004B, 
                     0 0 20px #00004B, 0 0 20px #00004B;
        margin-bottom: 0px;
    }
    </style>
    """

    # Inject CSS for glowing title
    st.markdown(glowing_title, unsafe_allow_html=True)

    # Glowing title
    st.markdown('<h1 class="glow">GALACTIC X</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="text-align: center; font-size: 30px;"><b>Welcome to GALACTIC-X! Your all-in-one tool for advanced galaxy data analysis.</b></p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h1 style="text-align: center; font-size: 50px;">What do we Provide?</h1>
        """,
        unsafe_allow_html=True,
    )

    # Create four columns for the description
    col1, col2, col3, col4 = st.columns(4)

    with col1:

        with st.container(border=True, height=200):
            first_co, second_co, third_co, fourth_co, fifth_co = st.columns(5)
            with third_co:
                st.image(
                    "classification.png",
                    width=30,
                )
            st.markdown(
                """
            <div style="text-align: center; font-weight: bold;">
                <p style="font-size:20px"><b>Dynamically predict galaxy types, redshift values, or star formation rates using machine learning. The model adjusts predictions based on updated astronomical observations, improving accuracy over time</b></p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            # st.image("https://picsum.photos/200/300")
        # st.markdown(
        #     """
        #     <div style="text-align: center; font-weight: bold;">
        #         <p>Dynamically predict galaxy types, redshift values, or star formation rates using machine learning. The model adjusts predictions based on updated astronomical observations, improving accuracy over time</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )

    with col2:
        # first_co, second_co, third_co, fourth_co, fifth_co = st.columns(5)
        # with third_co:
        #     st.image(
        #         "clustering.png",
        #         width=30,
        #     )
        # st.markdown(
        #     """
        #     <div style="text-align: center; font-weight: bold;">
        #         <p>Utilize machine learning algorithms to classify galaxies into types such as spiral, elliptical, and irregular based on key features like morphology, brightness, and color. The model continuously refines classifications as new data becomes available, improving accuracy and insight into galactic formation and evolution.</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )
        with st.container(border=True, height=200):
            first_co, second_co, third_co, fourth_co, fifth_co = st.columns(5)
            with third_co:
                st.image(
                    "clustering.png",
                    width=30,
                )
            # st.write(
            #     "##### Utilize machine learning algorithms to classify galaxies into types such as spiral, elliptical, and irregular based on key features like morphology, brightness, and color. The model continuously refines classifications as new data becomes available, improving accuracy and insight into galactic formation and evolution. #####"
            # )
            st.markdown(
                """
            <div style="text-align: center; font-weight: bold;">
                <p style="font-size:20px"><b>Utilize machine learning algorithms to classify galaxies into types such as spiral, elliptical, and irregular based on key features like morphology, brightness, and color. The model continuously refines classifications as new data becomes available, improving accuracy and insight into galactic formation and evolution.</b></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col3:
        # first_co, second_co, third_co, fourth_co, fifth_co = st.columns(5)
        # with third_co:
        #     st.image(
        #         "globular.png",
        #         width=30,
        #     )
        # st.markdown(
        #     """
        #     <div style="text-align: center; font-weight: bold;">
        #         <p>Apply clustering and classification algorithms and photometric analysis to detect and categorize globular clusters in galaxies. Use advanced machine learning techniques to assess their properties to provide deeper insights.</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )
        with st.container(border=True, height=200):
            first_co, second_co, third_co, fourth_co, fifth_co = st.columns(5)
            with third_co:
                st.image(
                    "globular.png",
                    width=30,
                )
            # st.write(
            #     "##### Apply clustering and classification algorithms and photometric analysis to detect and categorize globular clusters in galaxies. Use advanced machine learning techniques to assess their properties to provide deeper insights. #####"
            # )
            st.markdown(
                """
            <div style="text-align: center; font-weight: bold;">
                <p style="font-size:20px"><b>Apply clustering and classification algorithms and photometric analysis to detect and categorize globular clusters in galaxies. Use advanced machine learning techniques to assess their properties to provide deeper insights.</b></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col4:
        # first_co, second_co, third_co, fourth_co, fifth_co = st.columns(5)
        # with third_co:
        #     st.image(
        #         "enhancer.png",
        #         width=30,
        #     )
        # st.markdown(
        #     """
        #     <div style="text-align: center; font-weight: bold;">
        #         <p>Leverage neural networks such as Generative Adversarial Networks (GANs) to upscale low-resolution galaxy images. This system enhances image clarity and detail, making it easier to identify faint astronomical objects and analyze galaxy structures with greater precision.</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )
        with st.container(border=True, height=200):
            first_co, second_co, third_co, fourth_co, fifth_co = st.columns(5)
            with third_co:
                st.image(
                    "enhancer.png",
                    width=30,
                )
            # st.write(
            #     "##### Leverage neural networks such as Generative Adversarial Networks (GANs) to upscale low-resolution galaxy images. This system enhances image clarity and detail, making it easier to identify faint astronomical objects and analyze galaxy structures with greater precision. #####"
            # )
            st.markdown(
                """
            <div style="text-align: center; font-weight: bold;">
                <p style="font-size:20px"><b>Leverage neural networks such as Generative Adversarial Networks (GANs) to upscale low-resolution galaxy images. This system enhances image clarity and detail, making it easier to identify faint astronomical objects and analyze galaxy structures with greater precision.</b></p>
            </div>
            """,
                unsafe_allow_html=True,
            )


# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Galaxy Classifier & Redshift Predictor",
        "Galaxy Image Resolution Enhancer",
        "Globular Cluster Analysis",
        "Galaxy Cluster Analysis",
        "Chatbot",
    ],
)

# Display content based on the selected option
if app_mode == "Home":
    # Add background image to the home page
    add_bg_from_local("bg-7.jpg")  # Replace with your image file
    home_page()

elif app_mode == "Galaxy Classifier & Redshift Predictor":
    add_bg_from_local("thrinith2.jpg")
    galaxy_classifier()

elif app_mode == "Galaxy Image Resolution Enhancer":
    add_bg_from_local("thrinith2.jpg")
    image_enhancer()

elif app_mode == "Globular Cluster Analysis":
    add_bg_from_local("amasha2.jpg")
    globular_cluster_analysis()

elif app_mode == "Galaxy Cluster Analysis":
    add_bg_from_local("amasha2.jpg")
    galaxy_cluster_analysis()

elif app_mode == "Chatbot":
    astro_chatbot()
