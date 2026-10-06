import os

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================


st.write("RUNNING APP FILE:", os.path.abspath(__file__))
st.set_page_config(
    page_title="Municipal Water Reliability Predictor",
    page_icon="💧",
    layout="centered"
)


# ============================================================
# LOAD SAVED MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "final_water_reliability_pipeline.joblib"
    )


try:
    model = load_model()

except Exception as error:
    st.error("The prediction model could not be loaded.")

    st.write("Actual error:")
    st.code(str(error))

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title(
    "💧 Municipal Water Reliability Predictor"
)

st.write(
    """
    This application estimates whether a Water Services Authority
    (WSA) is likely to have **reliable** or **low-reliability**
    water supply based on population and water-access information.
    """
)


st.info(
    """
    **Class guidance**

    Class 0: Reliable water supply

    Class 1: Low-reliability water supply
    """
)


# ============================================================
# USER INPUT
# ============================================================

st.subheader(
    "Enter WSA information"
)



population = st.number_input(
    "Total population",
    min_value=1.0,
    value=100000.0,
    step=1000.0
)


population_access = st.number_input(
    "Population with access to water",
    min_value=0.0,
    value=80000.0,
    step=1000.0
)


urban_population = st.number_input(
    "Urban population",
    min_value=0.0,
    value=60000.0,
    step=1000.0
)


urban_access = st.number_input(
    "Urban population with access to water",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)


rural_population = st.number_input(
    "Rural population",
    min_value=0.0,
    value=40000.0,
    step=1000.0
)


rural_access = st.number_input(
    "Rural population with access to water",
    min_value=0.0,
    value=30000.0,
    step=1000.0
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "Predict Water Reliability",
    type="primary"
):

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    errors = []


    if population_access > population:

        errors.append(
            "Population with access to water cannot exceed "
            "the total population."
        )


    if urban_access > urban_population:

        errors.append(
            "Urban population with access to water cannot "
            "exceed the urban population."
        )


    if rural_access > rural_population:

        errors.append(
            "Rural population with access to water cannot "
            "exceed the rural population."
        )


    if urban_population + rural_population > population:

        errors.append(
            "Urban population plus rural population cannot "
            "exceed the total population."
        )


    # --------------------------------------------------------
    # DISPLAY ERRORS
    # --------------------------------------------------------

    if errors:

        st.error(
            "Please correct the following input:"
        )

        for error in errors:

            st.write(
                f"- {error}"
            )


    else:

        # ====================================================
        # FEATURE ENGINEERING
        # ====================================================

        population_access_rate = (
            population_access
            / population
        ) * 100


        if urban_population > 0:

            urban_access_rate = (
                urban_access
                / urban_population
            ) * 100

        else:

            urban_access_rate = np.nan


        if rural_population > 0:

            rural_access_rate = (
                rural_access
                / rural_population
            ) * 100

        else:

            rural_access_rate = np.nan


        urban_population_share = (
            urban_population
            / population
        ) * 100


        rural_population_share = (
            rural_population
            / population
        ) * 100


        # ====================================================
        # CREATE MODEL INPUT
        # ====================================================

        input_data = pd.DataFrame({

            "Population": [
                population
            ],

            "Population with access to water": [
                population_access
            ],

            "Urban Population": [
                urban_population
            ],

            "Urban Population with access to water": [
                urban_access
            ],

            "Rural Population": [
                rural_population
            ],

            "Rural Population with access to water": [
                rural_access
            ],

            "population_access_rate": [
                population_access_rate
            ],

            "urban_access_rate": [
                urban_access_rate
            ],

            "rural_access_rate": [
                rural_access_rate
            ],

            "urban_population_share": [
                urban_population_share
            ],

            "rural_population_share": [
                rural_population_share
            ]
        })


        # ====================================================
        # RUN SAVED PIPELINE
        # ====================================================

        prediction = model.predict(
            input_data
        )[0]


        probabilities = model.predict_proba(
            input_data
        )[0]


        low_reliability_probability = (
            probabilities[1]
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        if prediction == 1:

            st.error(
                "Prediction: LOW-RELIABILITY WATER SUPPLY"
            )

            st.write(
                """
                The model predicts that this WSA is more likely
                to belong to the low-reliability class.
                """
            )

        else:

            st.success(
                "Prediction: RELIABLE WATER SUPPLY"
            )

            st.write(
                """
                The model predicts that this WSA is more likely
                to belong to the reliable water-supply class.
                """
            )


        st.metric(
            "Probability of Low Reliability",
            f"{low_reliability_probability:.1%}"
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        with st.expander(
            "View calculated features"
        ):

            st.write(
                f"Overall water-access rate: "
                f"{population_access_rate:.1f}%"
            )

            if not np.isnan(
                urban_access_rate
            ):

                st.write(
                    f"Urban water-access rate: "
                    f"{urban_access_rate:.1f}%"
                )

            if not np.isnan(
                rural_access_rate
            ):

                st.write(
                    f"Rural water-access rate: "
                    f"{rural_access_rate:.1f}%"
                )

            st.write(
                f"Urban population share: "
                f"{urban_population_share:.1f}%"
            )

            st.write(
                f"Rural population share: "
                f"{rural_population_share:.1f}%"
            )


# ============================================================
# RESPONSIBLE USE
# ============================================================

st.divider()

st.subheader(
    "Responsible Use and Limitations"
)

st.warning(
    """
    This model is an academic decision-support prototype and should
    not be used as the sole basis for municipal planning, funding,
    service-delivery decisions, or decisions affecting communities.

    Predictions are based on patterns in the available training data
    and may be uncertain for municipalities that differ substantially
    from the training observations. Human review and current official
    water-service information should be used alongside the prediction.
    """
)

st.caption(
    "IDA117V Introduction to Data Science Group Project"
)