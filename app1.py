import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model
MODEL_FILENAME = "model.sv"
model = pickle.load(open(MODEL_FILENAME, 'rb'))

# Mapping dictionaries
PCLASS_MAP = {0: "First Class", 1: "Second Class", 2: "Third Class"}
EMBARKED_MAP = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}
SEX_MAP = {0: "Female", 1: "Male"}

# Correct feature names to match the training dataset
FEATURE_NAMES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


# Streamlit app definition
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Titanic Survival Prediction",
        page_icon="🚢",
        layout="wide"
    )

    # App header
    st.markdown(
        """
        <style>
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="title">Titanic Survival Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Will you survive the Titanic disaster? Let’s find out!</div>',
                unsafe_allow_html=True)

    # Columns for input
    with st.container():
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image("https://media.gettyimages.com/id/520112444/photo/forepeek-of-titanic-shipwreck.jpg?s=612x612&w=gi&k=20&c=rgIlHYkBU1bBF4Kb6hli9Mz0Fpez9VVc_7c3Ni1FCzs=",
                     caption="The RMS Titanic", use_column_width=True)

        with col2:
            st.subheader("Input Passenger Details")
            sex_radio = st.radio("Gender", list(SEX_MAP.keys()), format_func=lambda x: SEX_MAP[x], index=1)
            pclass_radio = st.radio("Ticket Class", list(PCLASS_MAP.keys()), format_func=lambda x: PCLASS_MAP[x],
                                    index=2)
            embarked_radio = st.radio("Port of Embarkation", list(EMBARKED_MAP.keys()), index=2,
                                      format_func=lambda x: EMBARKED_MAP[x])

    st.markdown("---")

    with st.container():
        col3, col4, col5 = st.columns(3)

        with col3:
            age_slider = st.slider("Age", min_value=0, max_value=80, value=30, step=1)

        with col4:
            sibsp_slider = st.slider("Number of Siblings/Spouse", min_value=0, max_value=10, value=1, step=1)

        with col5:
            parch_slider = st.slider("Number of Parents/Children", min_value=0, max_value=6, value=0, step=1)

        fare_slider = st.slider("Ticket Fare ($)", min_value=0, max_value=512, value=100, step=1)

    # Prepare input data with matching feature names
    input_data = [[pclass_radio, sex_radio, age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    data = pd.DataFrame(input_data, columns=FEATURE_NAMES)

    # Model prediction
    survival = model.predict(data)[0]
    confidence = model.predict_proba(data)[0][survival] * 100

    # Prediction result
    st.markdown("---")
    st.subheader("Prediction Results")
    result_text = "Yes, you would survive! 🎉" if survival == 1 else "No, unfortunately you would not survive. 😔"
    result_color = "green" if survival == 1 else "red"

    st.markdown(
        f"""
        <style>
        .result {{
            font-size: 1.5rem;
            text-align: center;
            color: {result_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f'<div class="result">{result_text}</div>', unsafe_allow_html=True)
    st.write(f"Prediction Confidence: **{confidence:.2f}%**")


if __name__ == "__main__":
    main()
