#%%
from enum import unique
from altair.vegalite.v4.schema.channels import Color
from altair.vegalite.v4.schema.core import Axis
from numpy.core.defchararray import title
import streamlit as st  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import altair as alt
import pydeck as pdk

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

#%%

st.title("Partial Least Squares Cis-DP Concentration Prediction by IR")
########## Generate the training model ##########

st.write("Version 1.0")
st.subheader("Nguyen D. Vu ")

# read in data and wavelength info:
# Read IN training Data and test data
#data = pd.read_csv("corrected_spectra.csv", header=None)
train_uploaded_csv = st.file_uploader("Upload The Training Spectra (X)")
if train_uploaded_csv is not None:
    data = pd.read_csv(train_uploaded_csv, header=None)
    wavelengths = (pd.read_csv("Wavenumber data.csv").columns).astype(float)
else:
    st.exception(exception=NameError("Import The training set to initiate model"))
    st.stop()

#Get X matrix and y
y_uploaded_csv = st.file_uploader("Upload The Expected Concentration (Y)")
try:    
    y = pd.read_csv(y_uploaded_csv, header=None)[0].values
except:
    st.write("Import The training set to initiate model")
    st.stop()

X = data.values


st.subheader("Training Set Concentration Distribution (mg/mL)")
y_plot = pd.Series(y, name="mg/mL")
conc_distr = sns.displot(y_plot, bins=12, discrete=True)
st.pyplot(conc_distr)


########## PLOT WAVELENGTHS ##############
st.subheader("Training Model Spectras")
spectras = plt.figure(figsize=(14, 6.5))
with plt.style.context('seaborn-notebook'):
    plt.plot(wavelengths, X.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Abs")

st.write(spectras)

# window length = 19
win_len = st.sidebar.slider("Select window length for Savitzky-Golay norm", 7, 25, step=2)
poly_order = st.sidebar.slider("Select Polynomial fit for Savitzky-Golay norm", 1, 5, step=1)
deriv_ = st.sidebar.slider("Select derivative for Savitzky-Golay norm", 0, 4, step=1)
X2 = savgol_filter(X, win_len, polyorder=poly_order, deriv=deriv_)

st.subheader("Savitzky-Golay Normalized Training Model Spectras")
normalized_spectras = plt.figure(figsize=(14, 6.5))
with plt.style.context('seaborn-notebook'):
    plt.plot(wavelengths, X2.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Normalized Abs")

st.write(normalized_spectras)

cv_ = st.sidebar.slider(" K-Fold Validation, K = ", 3, 15, step=1)

########## Training Model Functions #####################

def optimise_pls_cv(X, y, n_comp):
    # Define PLS object
    pls = PLSRegression(n_components=n_comp)

    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=cv_)

    # Calculate scores
    r2 = r2_score(y, y_cv)
    mse = mean_squared_error(y, y_cv)
    rpd = y.std()/np.sqrt(mse)
    
    return (y_cv, r2, mse, rpd)

# test with 30 components
r2s = []
mses = []
rpds = []
xticks = np.arange(1, 30)
for n_comp in xticks:
    y_cv, r2, mse, rpd = optimise_pls_cv(X2, y, n_comp)
    r2s.append(r2)
    mses.append(mse)
    rpds.append(rpd)

# Plot the mses

def plot_metrics(vals, ylabel, objective):
    plot_out = plt.figure(figsize=(14, 6.5))
    with plt.style.context('fivethirtyeight'):
        plt.plot(xticks, np.array(vals), '-v', color='green', mfc='green')
        if objective=='min':
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], '*', ms=20, mfc='red')

        plt.xlabel('Number of PLS components')
        plt.xticks = xticks
        plt.ylabel(ylabel)
    return plot_out

st.subheader("Mean Square Error")
mse_plot = plot_metrics(mses, "MSE", "min")
st.pyplot(mse_plot)

st.subheader("Residual Predicion Deviation")
rpd_plot = plot_metrics(rpds, 'RPD', 'max')
st.pyplot(rpd_plot)

st.subheader("Coefficient of Determination (R-Squared)")
r2_plot = plot_metrics(r2s, 'R2', 'max')
st.pyplot(r2_plot)


st.subheader("Examine the 3 diagnostic plot above and enter the optimal number of components")
opt_nComp = st.number_input("Optimal # of components =", min_value=1, max_value=17, step=1)

y_cv, r2, mse, rpd = optimise_pls_cv(X2, y, opt_nComp)

st.write('R2 = %0.4f, MSE = %0.4f, RPD = %0.4f' %(r2, mse, rpd))

#plot final model
final_model = plt.figure(figsize=(6, 4))
with plt.style.context('seaborn'):
    plt.scatter(y, y_cv, color='black', s=10)
    plt.plot(y, y, color='red', label='Model')
    z = np.polyfit(y, y_cv, 1)
    plt.plot(np.polyval(z, y), y, color='blue', label='Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.plot()

st.pyplot(final_model)

## Upload test file
uploaded_csv = st.file_uploader("Upload your spectra for prediction")
if uploaded_csv is not None:
    test_data = pd.read_csv(uploaded_csv, header=None)

    #get X test matrix
    X_test = test_data.values[:, :]
    X_test2 = savgol_filter(X_test,win_len, polyorder=poly_order, deriv=deriv_) 
    pls = PLSRegression(n_components=opt_nComp)

    pls_model = pls.fit(X2, y)

    y_pred = pls_model.predict(X_test2)
    y_pred_df = pd.DataFrame({"predicted_mg/mL":(y_pred.reshape(1, -1)).flatten()})
    y_pred_chart = alt.Chart(y_pred_df).mark_bar().encode(
        alt.X("predicted_mg/mL", bin=True),
        y='count()',
    )
    st.altair_chart(y_pred_chart, use_container_width=True)
    st.write("Predicted Mean Concentration = {:.2f} mg/mL   \nStandard Deviation = {:.2f} mg/mL".format(np.mean(y_pred), np.std(y_pred)))
    #st.write("Expected Concentration by HPLC = 9.6 mg/mL")

    st.table(pd.DataFrame({"Predicted mg/mL":(y_pred.flatten())}))

# %%
