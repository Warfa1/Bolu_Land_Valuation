# A GIS and ML-Based Land Parcel Valuation System for Decision Support

**Course:** CME434 Geographic Information Systems, Karabük University
**Supervisor:** Dr. Öğr. Üyesi Sohaib K. M. Abujayyab

## Group Members
*   [Your Name] - [Your Student Number]
*   [Friend Name 1] - [Friend Student Number]
*   [Friend Name 2] - [Friend Student Number]

## Hosted Web GIS Application
[Link will be added here after Streamlit deployment]

## Project Overview
This project estimates land value (TL/m²) in Bolu, Türkiye, using a spatial regression dataset with 33 features (Ad details, Remote Sensing from GEE, and OSM accessibility). We compared Random Forest, XGBoost, and LightGBM using Spatial Block Cross-Validation, tuned the models, and explained the best model (XGBoost) with SHAP. The results are published as this open-source repository and a hosted Streamlit web app.

## Repository Contents
- `app.py`: The Streamlit Decision Support System (DSS) application.
- `requirements.txt`: The required Python libraries to run the app.
- `regression_dataset.csv`: The spatial regression dataset (features and target label).
- `best_model.pkl`: The trained XGBoost model used by the app.

## Data Dictionary
*(See the `data_dictionary.md` file in this repo for full feature descriptions.)*
