"""Streamlit interface for the corrected locked water-reliability pipeline."""
from pathlib import Path
import logging

import streamlit as st

from water_reliability.constants import SAFE_RAW_FEATURES
from water_reliability.inference import load_pipeline, predict, validate_inputs

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'artifacts' / 'water_reliability_pipeline.joblib'
LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title='Water Reliability Predictor', page_icon='💧', layout='centered')
st.title('💧 Water Reliability Predictor')
st.write('Estimate the reported water-supply reliability category of a Water Services Authority (WSA) from population and water-access counts.')
st.info('Class 1: reliable supply below 60%. Class 0: at least 60%. This team cutoff describes cross-sectional conditions; it does not forecast future outages.')
st.caption('Enter counts of people, not percentages. Urban and rural totals must reconcile. A zero urban or rural population is allowed.')

@st.cache_resource
def get_model():
    return load_pipeline(MODEL_PATH)

try:
    model = get_model()
except Exception:
    LOGGER.exception('Unable to load the packaged water-reliability pipeline')
    st.error('The prediction model is unavailable. Ask the administrator to check the model bundle and install the pinned requirements.')
    st.stop()

values = {}
defaults = [100000.0, 80000.0, 60000.0, 50000.0, 40000.0, 30000.0]
for name, initial in zip(SAFE_RAW_FEATURES, defaults):
    values[name] = st.number_input(name, min_value=0.0, value=initial, step=1000.0, format='%.0f')

if st.button('Predict Water Reliability', type='primary'):
    try:
        validate_inputs(values)
    except ValueError as error:
        st.error(str(error))
    else:
        try:
            predicted, probability = predict(model, values)
        except Exception:
            LOGGER.exception('Water-reliability prediction failed')
            st.error('The prediction could not be completed. Please try again or contact the administrator.')
        else:
            st.subheader('Prediction result')
            reliable_probability = 1.0 - probability
            category = 'UNRELIABLE' if predicted == 1 else 'RELIABLE'
            color = '#ef4444' if predicted == 1 else '#16a34a'
            st.html(f'''
                <div style="border: 1px solid #80808040; border-radius: 12px; padding: 16px;">
                    <span class="prediction-badge" style="display: inline-block; color: {color}; background-color: {color}20; border-radius: 999px; padding: 2px 8px; font-size: 0.8rem; font-weight: 600;">Predicted: {category}</span>
                    <div style="margin-top: 8px; font-size: 0.875rem; opacity: 0.75;">Probability of Reliable</div>
                    <div class="reliable-probability" style="color: {color}; font-size: 2.5rem; font-weight: 600; margin: 4px 0;">{reliable_probability:.1%}</div>
                    <div role="progressbar" aria-label="Probability of Reliable" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{reliable_probability * 100}" aria-valuetext="{reliable_probability:.1%}" style="height: 12px; background-color: #80808030; border-radius: 999px; overflow: hidden;">
                        <div style="width: {reliable_probability:.10%}; height: 100%; background-color: {color}; border-radius: 999px;"></div>
                    </div>
                    <div class="unreliable-probability" style="display: flex; justify-content: space-between; gap: 8px; margin-top: 10px; font-size: 0.875rem;">
                        <span style="opacity: 0.75;">Probability of Unreliable</span><strong>{probability:.1%}</strong>
                    </div>
                </div>
            ''')
            st.caption('These are model class probabilities, not the reliable-supply percentage. Water access alone does not measure reliable supply.')
            st.write('Use this result to support further investigation with current official information. A reliable prediction does not guarantee uninterrupted water for every resident.')
            st.caption('This probability is a model estimate and has not been independently calibrated. It is not the percentage of residents receiving reliable water.')

st.divider()
st.warning('Academic decision-support prototype. Do not use predictions alone for funding, municipal penalties or operational decisions. The small dataset, source inconsistencies and subgroup errors limit generalisation; human review is required.')
st.caption('IDA117V Introduction to Data Science Group Project')
