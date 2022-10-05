# Load packages (comments for more special stuff)

import pandas as pd
import pickle # un-pickling stuff from training notebook
from xgboost import XGBRegressor # we use a trained XGBoost model...and therefore need to load it
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import shap # add prediction explainability
import matplotlib.pyplot as plt

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


tab1, tab2, tab3, tab4 = st.tabs(["Predictor tool SML", "HR Managererial Tool", "SML", "Customer Segmentation and UML"])

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
        with st.expander("Recommended customer segmentation based on plots"):
            'Here we will perform EDA based on our bank-client data and use this to recommend specific groups of customers'
            



        with st.expander("UML"):
            st.title("Unsupervised Machine Learning")
            st.subheader('This will be a journey through the creation of UML customer segmentation, and an analysis of the obtained result.')
            'Let us start with the end result'
            st.image('https://raw.githubusercontent.com/Ceges98/BDS-Project/main/visualization.png', caption='not an optimal result')
            st.subheader('How did this come to be?')
            'To start the process of customer segmentation we need data regarding them.'
            data_raw = data.iloc[:, 0:7]
            st.write(data_raw.head(100))
            st.caption('these are the first 100 entrances in our relevant dataset, currently unfiltered.')
            'Some work is needed for this data to be operable in regards to UML, first we remove the unknown'
            data_raw = data_raw[data_raw["job"].str.contains("unknown") == False]
            data_raw = data_raw[data_raw["marital"].str.contains("unknown") == False]
            data_raw = data_raw[data_raw["education"].str.contains("unknown") == False]
            data_raw = data_raw[data_raw["housing"].str.contains("unknown") == False]
            data_raw = data_raw[data_raw["loan"].str.contains("unknown") == False]
            data_raw.drop('default', inplace=True, axis=1)
            data = data_raw
            tab01, tab02 = st.tabs(['new data', 'code'])
            with tab01:
                st.write(data_raw.head(50))
                st.caption('now there are no unknown values, we have also dropped the default column as it is almost solely "no" values and therefore should not be used to segment the customers.')
            with tab02:
                drop_unknown = '''data_raw = data_raw[data_raw["job"].str.contains("unknown") == False]
            data_raw = data_raw[data_raw["marital"].str.contains("unknown") == False]
            data_raw = data_raw[data_raw["education"].str.contains("unknown") == False]
            data_raw = data_raw[data_raw["housing"].str.contains("unknown") == False]
            data_raw = data_raw[data_raw["loan"].str.contains("unknown") == False]
            data_raw.drop('default', inplace=True, axis=1)''' 
                st.code(drop_unknown, language='python')
            'Next up is the fact that our data is unusable due to it being in a non-numerical format'
            'To fix this spread age out into 4 categories, replace yes/no with 1/0 on housing/loan, LabelEncode education and make a, admittedly subjective, list for jobs based on income'
            def age(data_raw):
                data_raw.loc[data_raw['age'] <= 30, 'age'] = 1
                data_raw.loc[(data_raw['age'] > 30) & (data_raw['age'] <= 45), 'age'] = 2
                data_raw.loc[(data_raw['age'] > 45) & (data_raw['age'] <= 65), 'age'] = 3
                data_raw.loc[(data_raw['age'] > 65) & (data_raw['age'] <= 98), 'age'] = 4 
                return data_raw
            age(data_raw);
            data_raw = data_raw.replace(to_replace=['yes', 'no'], value=[1, 0])
            data_raw = data_raw.replace(to_replace=['unemployed', 'student', 'housemaid', 'blue-collar', 'services', 'retired', 'technician', 'admin.', 'self-employed', 'entrepreneur', 'management'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            labelencoder_X = LabelEncoder()  
            data_raw['education'] = data_raw['education'].replace({'illiterate':'a_illiterate'})
            data_raw['education'] = labelencoder_X.fit_transform(data_raw['education'])
            tab03, tab04 = st.tabs(['numeric data', 'code'])
            with tab03:
                st.write(data_raw.head(50))
            
            with tab04:
                numerification = '''def age(data_raw):
                data_raw.loc[data_raw['age'] <= 30, 'age'] = 1
                data_raw.loc[(data_raw['age'] > 30) & (data_raw['age'] <= 45), 'age'] = 2
                data_raw.loc[(data_raw['age'] > 45) & (data_raw['age'] <= 65), 'age'] = 3
                data_raw.loc[(data_raw['age'] > 65) & (data_raw['age'] <= 98), 'age'] = 4 
                return data_raw
            age(data_raw);
            data_raw = data_raw.replace(to_replace=['yes', 'no'], value=[1, 0])
            data_raw = data_raw.replace(to_replace=['unemployed', 'student', 'housemaid', 'blue-collar', 'services', 'retired', 'technician', 'admin.', 'self-employed', 'entrepreneur', 'management'], value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            labelencoder_X = LabelEncoder()  
            data_raw['education'] = data_raw['education'].replace({'illiterate':'a_illiterate'})
            data_raw['education'] = labelencoder_X.fit_transform(data_raw['education'])'''
                st.code(numerification, language='python')
            st.caption('this is not a perfect way of handling the issue but onehotencoding gave rise to different issues.')
            'It may be noted that marriage is currently untouched, this is due to troubles with OneHotEncoding. As such is was deemed unwise to throw in yet another subjective variable. It will therefor be dropped.'
            data_raw = data_raw.drop(columns = 'marital')
            st.write(data_raw.head())
            'lastly these numbers need to be scaled'
            data_raw_scaled = scaler.fit_transform(data_raw)
            tab05, tab06 = st.tabs(['scaled data', 'code'])
            with tab05:
                st.write(data_raw_scaled[:10])
            with tab06:
                scaled_date = '''data_raw_scaled = scaler.fit_transform(data_raw)'''
                st.code(scaled_date, language='python')
            st.caption('Now the previous sizes of the values have been standard scaled.')
            'From here on out the process will be shown through code with comments'
            rest = '''#umap accepts standard-scaled data
embeddings = umap_scaler.fit_transform(data_raw_scaled)

#we choose 6 clusters
clusterer = KMeans(n_clusters=6)

Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_raw_scaled)
    Sum_of_squared_distances.append(km.inertia_)

#no clear elbow
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#we fit clusters on our scaled data
clusterer.fit(data_raw_scaled)

#we then copy the clusters into the original file
data['cluster'] = clusterer.labels_

#can use the clusters to fx. see the mean of age in our clusters.
#note that age does not seem a big factor in clustering as the mean is mostly the same.
data.groupby('cluster').age.mean()

#prepping our vis_data
vis_data = pd.DataFrame(embeddings)
vis_data['cluster'] = data['cluster']
vis_data['education'] = data['education']
vis_data['age'] = data['age']
vis_data['job'] = data['job']
vis_data['marital'] = data['marital']
vis_data['housing'] = data['housing']
vis_data['loan'] = data['loan']

vis_data.columns = ['x', 'y', 'cluster','education', 'age', 'job', 'marital', 'housing', 'loan']

#finally plotting the data with relevant tooltips
#for unknown reasons a null cluster is made alongside our other clusters
alt.data_transformers.enable('default', max_rows=None)
alt.Chart(vis_data).mark_circle(size=60).encode(
    x='x',
    y='y',
    tooltip=['education', 'age', 'job', 'marital', 'housing', 'loan'],
    color=alt.Color('cluster:N', scale=alt.Scale(scheme='dark2')) #use N after the var to tell altair that it's categorical
).interactive()'''
            st.code(rest, language='python')
            'The reasoning behind showing this block of code is mainly to show the procedure that was taken following the data-preprocessing and showing a more in-depth process is not very useful as the end result is flawed.'
            'Speaking of, here we have once again the result so that the flaws can be discussed'
            st.image('https://raw.githubusercontent.com/Ceges98/BDS-Project/main/visualization.png', caption='still not optimal')
            '''To understand the flaws we have to look at the goal of the model. 
            The goal of this model was to place the customers in clusters based on their data.
            As such there are two problems:
            1. The clusters are randomly dispersed.
            2. An extra null-cluster has been created.
            Optimally we would be able to find and fix the problem causing these flaws but as of know this model
            has presented a learning opportunity and not a finished piece of work.'''
