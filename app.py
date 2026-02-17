import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Saved Model
# -----------------------------
model = joblib.load("employee_retention_model.pkl")

st.set_page_config(page_title="Employee Retention Predictor", layout="wide")

st.title("🔍 Employee Job Change Prediction")
st.write("Fill the employee details below to predict job change risk.")

# -----------------------------
# USER INPUTS
# -----------------------------

city = st.selectbox(
    "City",
    sorted(['city_1',
 'city_10',
 'city_100',
 'city_101',
 'city_102',
 'city_103',
 'city_104',
 'city_105',
 'city_106',
 'city_107',
 'city_109',
 'city_11',
 'city_111',
 'city_114',
 'city_115',
 'city_116',
 'city_117',
 'city_118',
 'city_12',
 'city_120',
 'city_121',
 'city_123',
 'city_126',
 'city_127',
 'city_128',
 'city_129',
 'city_13',
 'city_131',
 'city_133',
 'city_134',
 'city_136',
 'city_138',
 'city_139',
 'city_14',
 'city_140',
 'city_141',
 'city_142',
 'city_143',
 'city_144',
 'city_145',
 'city_146',
 'city_149',
 'city_150',
 'city_152',
 'city_155',
 'city_157',
 'city_158',
 'city_159',
 'city_16',
 'city_160',
 'city_162',
 'city_165',
 'city_166',
 'city_167',
 'city_171',
 'city_173',
 'city_175',
 'city_176',
 'city_179',
 'city_18',
 'city_180',
 'city_19',
 'city_2',
 'city_20',
 'city_21',
 'city_23',
 'city_24',
 'city_25',
 'city_26',
 'city_27',
 'city_28',
 'city_30',
 'city_31',
 'city_33',
 'city_36',
 'city_37',
 'city_39',
 'city_40',
 'city_41',
 'city_42',
 'city_43',
 'city_44',
 'city_45',
 'city_46',
 'city_48',
 'city_50',
 'city_53',
 'city_54',
 'city_55',
 'city_57',
 'city_59',
 'city_61',
 'city_62',
 'city_64',
 'city_65',
 'city_67',
 'city_69',
 'city_7',
 'city_70',
 'city_71',
 'city_72',
 'city_73',
 'city_74',
 'city_75',
 'city_76',
 'city_77',
 'city_78',
 'city_79',
 'city_8',
 'city_80',
 'city_81',
 'city_82',
 'city_83',
 'city_84',
 'city_89',
 'city_9',
 'city_90',
 'city_91',
 'city_93',
 'city_94',
 'city_97',
 'city_98',
 'city_99'])
)

city_development_index = st.slider("City Development Index", 0.0, 1.0, 0.5)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

relevent_experience = st.selectbox(
    "Relevant Experience",
    ["No relevant experience", "Has relevant experience"]
)

enrolled_university = st.selectbox(
    "Enrolled University",
    ["no_enrollment", "Full time course", "Part time course"]
)

education_level = st.selectbox(
    "Education Level",
    ["Primary School", "High School", "Graduate", "Masters", "Phd"]
)

major_discipline = st.selectbox(
    "Major Discipline",
    ["STEM", "Business Degree", "Arts", "Humanities", "Other", "No Major"]
)

experience = st.number_input("Total Experience (Years)", 0, 25, 5)

company_size = st.selectbox(
    "Company Size",
    ["1-10", "10-49", "50-99", "100-500", "500-999",
     "1000-4999", "5000-9999", "10000-15000"]
)

company_type = st.selectbox(
    "Company Type",
    ["Pvt Ltd", "Funded Startup", "Public Sector",
     "Early Stage Startup", "NGO", "Other"]
)

last_new_job = st.number_input("Years Since Last Job Change", 0, 5, 1)

training_hours = st.number_input("Training Hours", 0, 300, 50)



# -----------------------------
# Convert company size to numeric (same logic as training)
# -----------------------------
size_mapping = {
    '1-10': 5,
    '10-49': 30,
    '50-99': 75,
    '100-500': 300,
    '500-999': 750,
    '1000-4999': 3000,
    '5000-9999': 7500,
    '10000-15000': 12500
}

company_size_num = size_mapping.get(company_size, np.nan)

# -----------------------------
# Feature Engineering (same as training)
# -----------------------------

def create_features(df):
    df = df.copy()

    df['exp_to_training_ratio'] = df['experience'] / (df['training_hours'] + 1)
    df['high_dev_city'] = (df['city_development_index'] > 0.8).astype(int)
    df['recent_job_change'] = (df['last_new_job'] <= 1).astype(int)

    df['exp_category'] = pd.cut(
        df['experience'],
        bins=[-1, 5, 10, 15, 25],
        labels=['Entry', 'Mid', 'Senior', 'Expert']
    ).astype(str)

    df['training_intensity'] = df['training_hours'] / (df['experience'] + 1)

    df['company_size_category'] = pd.cut(
        df['company_size'],
        bins=[0, 100, 1000, 5000, 15000],
        labels=['Small', 'Medium', 'Large', 'Enterprise']
    ).astype(str)

    return df

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Job Change Risk"):

    input_data = pd.DataFrame({
        'city' : [city],
        'city_development_index': [city_development_index],
        'gender': [gender],
        'relevent_experience': [relevent_experience],
        'enrolled_university' : [enrolled_university],
        'education_level': [education_level],
        'major_discipline': [major_discipline],
        'experience': [experience],
        'company_size': [company_size_num],
        'company_type': [company_type],
        'last_new_job': [last_new_job],
        'training_hours': [training_hours],
    })

    # Apply feature engineering
    input_data = create_features(input_data)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠ High Risk of Leaving (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk (Probability: {probability:.2%})")

    st.write("Predicted Probability:", round(probability, 4))