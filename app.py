from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

st.set_page_config(
    page_title="Diagnostic maladies des plantes",
    page_icon=":seedling:",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "plant_disease_dataset.csv"

NUMERIC_COLUMNS = ["leaf_length", "leaf_width", "stem_diameter"]
TARGET_COLUMN = "disease_type"


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in NUMERIC_COLUMNS:
        cleaned[column] = cleaned[column].astype(float)
        cleaned[column] = cleaned[column].fillna(cleaned[column].mean())
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned[column] = np.clip(cleaned[column], lower, upper)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    return clean_data(load_raw_data())


def encode_features(
    df: pd.DataFrame,
    pesticide_encoder: LabelEncoder,
    soil_encoder: OneHotEncoder,
    weather_encoder: OneHotEncoder,
) -> pd.DataFrame:
    features = df[NUMERIC_COLUMNS + ["soil_type", "weather", "pesticide"]].copy()
    features["pesticide"] = pesticide_encoder.transform(features["pesticide"])
    soil_encoded = soil_encoder.transform(features[["soil_type"]])
    weather_encoded = weather_encoder.transform(features[["weather"]])
    soil_columns = soil_encoder.get_feature_names_out(["soil_type"])
    weather_columns = weather_encoder.get_feature_names_out(["weather"])
    features = pd.concat(
        [
            features.drop(columns=["soil_type", "weather"]),
            pd.DataFrame(soil_encoded, columns=soil_columns, index=features.index),
            pd.DataFrame(weather_encoded, columns=weather_columns, index=features.index),
        ],
        axis=1,
    )
    ordered_columns = [
        "leaf_length",
        "leaf_width",
        "stem_diameter",
        "pesticide",
        *soil_columns,
        *weather_columns,
    ]
    return features[ordered_columns]


@st.cache_resource
def prepare_artifacts():
    data = load_clean_data()
    pesticide_encoder = LabelEncoder().fit(data["pesticide"])
    soil_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    soil_encoder.fit(data[["soil_type"]])
    weather_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    weather_encoder.fit(data[["weather"]])
    encoded = encode_features(data, pesticide_encoder, soil_encoder, weather_encoder)
    label_encoder = LabelEncoder().fit(data[TARGET_COLUMN])
    X_train, _, y_train, _ = train_test_split(
        encoded,
        label_encoder.transform(data[TARGET_COLUMN]),
        test_size=0.2,
        random_state=23,
    )
    scaler = MinMaxScaler().fit(X_train)
    return {
        "pesticide_encoder": pesticide_encoder,
        "soil_encoder": soil_encoder,
        "weather_encoder": weather_encoder,
        "feature_columns": encoded.columns.tolist(),
        "label_encoder": label_encoder,
        "scaler": scaler,
    }


@st.cache_resource
def load_models():
    with (BASE_DIR / "plant_diseases.pkl").open("rb") as handle:
        logistic_model = pickle.load(handle)
    with (BASE_DIR / "plant_diseases_best.pkl").open("rb") as handle:
        extra_trees_model = pickle.load(handle)
    return logistic_model, extra_trees_model


def format_probabilities(model, probabilities, label_encoder):
    class_labels = label_encoder.inverse_transform(model.classes_)
    return dict(zip(class_labels, probabilities))


def ensure_session_defaults(data: pd.DataFrame):
    if "leaf_length" not in st.session_state:
        averages = data[NUMERIC_COLUMNS].mean()
        st.session_state.leaf_length = float(round(averages["leaf_length"], 3))
        st.session_state.leaf_width = float(round(averages["leaf_width"], 3))
        st.session_state.stem_diameter = float(round(averages["stem_diameter"], 3))
        st.session_state.soil_type = data["soil_type"].mode().iat[0]
        st.session_state.weather = data["weather"].mode().iat[0]
        st.session_state.pesticide = data["pesticide"].mode().iat[0]


def inject_random_sample(data: pd.DataFrame):
    sample = data.sample(1, random_state=None).iloc[0]
    st.session_state.leaf_length = float(round(sample["leaf_length"], 3))
    st.session_state.leaf_width = float(round(sample["leaf_width"], 3))
    st.session_state.stem_diameter = float(round(sample["stem_diameter"], 3))
    st.session_state.soil_type = sample["soil_type"]
    st.session_state.weather = sample["weather"]
    st.session_state.pesticide = sample["pesticide"]


def main():
    st.title("Diagnostic maladies des plantes")
    st.caption("Comparez deux modeles entraine pour identifier blight, mildew et rust.")
    clean_data_df = load_clean_data()
    artifacts = prepare_artifacts()
    logistic_model, extra_trees_model = load_models()
    ensure_session_defaults(clean_data_df)

    with st.sidebar:
        st.header("Configuration")
        if st.button("Observation aleatoire"):
            inject_random_sample(clean_data_df)
        st.write("Ajustez les variables puis validez le formulaire.")

    with st.form("prediction_form"):
        col_left, col_right = st.columns(2)
        with col_left:
            st.number_input("Longueur de la feuille", min_value=0.0, step=0.1, key="leaf_length")
            st.number_input("Largeur de la feuille", min_value=0.0, step=0.1, key="leaf_width")
            st.number_input("Diametre de la tige", min_value=0.0, step=0.05, key="stem_diameter")
        with col_right:
            st.selectbox(
                "Type de sol",
                sorted(clean_data_df["soil_type"].unique()),
                key="soil_type",
            )
            st.selectbox(
                "Meteo",
                sorted(clean_data_df["weather"].unique()),
                key="weather",
            )
            st.selectbox(
                "Traitement pesticide",
                sorted(clean_data_df["pesticide"].unique()),
                key="pesticide",
            )
        submitted = st.form_submit_button("Lancer la prediction")

    if submitted:
        input_payload = {
            "leaf_length": st.session_state.leaf_length,
            "leaf_width": st.session_state.leaf_width,
            "stem_diameter": st.session_state.stem_diameter,
            "soil_type": st.session_state.soil_type,
            "weather": st.session_state.weather,
            "pesticide": st.session_state.pesticide,
        }
        input_df = pd.DataFrame([input_payload])
        encoded_input = encode_features(
            input_df,
            artifacts["pesticide_encoder"],
            artifacts["soil_encoder"],
            artifacts["weather_encoder"],
        )[artifacts["feature_columns"]]
        scaled_input = artifacts["scaler"].transform(encoded_input)

        logistic_prediction = int(logistic_model.predict(scaled_input)[0])
        logistic_probs = logistic_model.predict_proba(scaled_input)[0]
        extra_prediction = int(extra_trees_model.predict(encoded_input.values)[0])
        extra_probs = extra_trees_model.predict_proba(encoded_input.values)[0]

        label_encoder = artifacts["label_encoder"]
        logistic_label = label_encoder.inverse_transform([logistic_prediction])[0]
        extra_label = label_encoder.inverse_transform([extra_prediction])[0]

        logistic_map = format_probabilities(logistic_model, logistic_probs, label_encoder)
        extra_map = format_probabilities(extra_trees_model, extra_probs, label_encoder)
        ordered_labels = list(label_encoder.classes_)
        prob_rows = []
        for label in ordered_labels:
            prob_rows.append(
                {
                    "Maladie": label,
                    "Regression logistique": logistic_map.get(label, 0.0),
                    "ExtraTrees": extra_map.get(label, 0.0),
                }
            )
        prob_df = pd.DataFrame(prob_rows).set_index("Maladie")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Regression logistique",
                logistic_label,
                delta=f"{logistic_map.get(logistic_label, 0.0) * 100:.1f} % de confiance",
            )
        with col_b:
            st.metric(
                "ExtraTrees",
                extra_label,
                delta=f"{extra_map.get(extra_label, 0.0) * 100:.1f} % de confiance",
            )

        st.subheader("Probabilites par modele")
        st.dataframe(prob_df.style.format("{:.2%}"), use_container_width=True)

        st.subheader("Variables soumises")
        st.dataframe(input_df, use_container_width=True)



if __name__ == "__main__":
    main()
