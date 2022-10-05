# Load packages (comments for more special stuff)

import pandas as pd
import pickle # un-pickling stuff from training notebook
from xgboost import XGBRegressor # we use a trained XGBoost model...and therefore need to load it
from sklearn.preprocessing import StandardScaler
import shap # add prediction explainability

import numpy as np
import itertools # we need that to flatten ohe.categories_ into one list for columns
import streamlit as st
from streamlit_shap import st_shap # wrapper to display nice shap viz in the app

##Streamlit interface:
st.set_page_config(page_title='Bank Marketing Project',
                    page_icon="🐙",
                    layout='wide')

colT1,colT2 = st.columns([10,20])
with colT2:
   st.title('HR Managing Tool Project')


tab1, tab2, tab3, tab4 = st.tabs(["Predictor tool SML", "HR Managererial Tool", "SML", "UML"])

with tab1:

    st.subheader('Will this given costumer say yes?')

    #this is how you can add images e.g. from unsplash (or loca image file)
    #st.image('https://images.unsplash.com/photo-1577346895253-445f05a6670d?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxyYW5kb218MHx8fHx8fHx8MTY2NDY1MzMwNQ&ixlib=rb-1.2.1&q=80&utm_campaign=api-credit&utm_medium=referral&utm_source=unsplash_source&w=1080', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    # use this decorator (--> @st.experimental_singleton) and 0-parameters function to only load and preprocess once
    @st.experimental_singleton
    def read_objects():
        model_xgb = pickle.load(open('model_xgb.pkl','rb'))
        scaler = pickle.load(open('scaler.pkl','rb'))
        ohe = pickle.load(open('ohe.pkl','rb'))
        shap_values = pickle.load(open('shap_values.pkl','rb'))
        cats = list(itertools.chain(*ohe.categories_))
        return model_xgb, scaler, ohe, cats, shap_values

    model_xgb, scaler, ohe, cats, shap_values = read_objects()

    # define explainer
    explainer = shap.TreeExplainer(model_xgb)

    #write some markdown blah
    with st.expander("What's that app?"):
        st.markdown("""
        This app will help you determine what you should be asking people to pay per night for staying at your awesome place.
        We trained an AI on successful places in Copenhagen. It will give you a pricing suggestion given a few inputs.
        We recommend going around 350kr up or down depending on the amenities that you can provide and the quality of your place.
        As a little extra 🌟, we added an AI explainer 🤖 to understand factors driving prices up or down.
        """)

    st.subheader('Costumer description')

    # here you collect all inputs from the user into new objects
    job = st.selectbox('What is his/hers jobtype?', options=ohe.categories_[0])
    marital = st.radio('Marital', options=ohe.categories_[1])
    poutcome = st.selectbox('What was the previous outcome for this costumer?', options=ohe.categories_[4])
    age = st.number_input('Age?', min_value=17, max_value=98)
    education = st.number_input('Education', min_value=0, max_value=7)
    campaign = st.number_input('How many contacts have you made for this costumer for this campagin already?', min_value=0, max_value=35)
    previous = st.number_input('How many times have you contacted this client before?', min_value=0, max_value=35)

    # make a nice button that triggers creation of a new data-line in the format that the model expects and prediction
    if st.button('Predict! 🚀'):
        # make a DF for categories and transform with one-hot-encoder
        new_df_cat = pd.DataFrame({'job':job,
                    'marital':marital,
                    'month': 'oct',
                    'day_of_week':'fri',
                    'poutcome':poutcome}, index=[0])
        new_values_cat = pd.DataFrame(ohe.transform(new_df_cat), columns = cats , index=[0])

        # make a DF for the numericals and standard scale
        new_df_num = pd.DataFrame({'age':age, 
                                'education': education,
                                'campaign': campaign,
                                'previous': previous, 
                                'emp.var.rate': 1.1,
                                'cons.price.idx': 93.994,
                                'cons.conf.idx': -36.4,
                                'euribor3m': 4.857,
                                'nr.employed': 5191.0
                            }, index=[0])
        new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns = new_df_num.columns, index=[0])  
        
        #bring all columns together
        line_to_pred = pd.concat([new_values_num, new_values_cat], axis=1)

        #run prediction for 1 new observation
        predicted_value = model_xgb.predict(line_to_pred)[0]

        #print out result to user
        st.metric(label="Predicted answer", value=f'{predicted_value}')
        
        #print SHAP explainer to user
        st.subheader(f'Why {predicted_value}? See below:')
        shap_value = explainer.shap_values(line_to_pred)
        st_shap(shap.force_plot(explainer.expected_value, shap_value, line_to_pred), height=400, width=500)

    with tab2:
        st.title("this is tab 2")
    with tab4:
        data = pd.read_csv("https://raw.githubusercontent.com/Ceges98/BDS-Project/main/bank_marketing.csv", sep=";")
        st.title("Unsupervised Machine Learning")
        st.subheader('This will be a journey through the creation of UML customer segmentation, and an analysis of the obtained result.')
        'Let us start with the end result'
        st.image('https://raw.githubusercontent.com/Ceges98/BDS-Project/main/visualization.png', caption='not an optimal result')
        st.subheader('How did this come to be?')
        'To start the process of customer segmentation we need data regarding them.'
        client_data = data.iloc[:, 0:7]
        st.write(client_data.head(100))
        st.caption('these are the first 100 entrances in our relevant dataset, currently unfiltered.')
        'Some work is needed for this data to be operable in regards to UML'