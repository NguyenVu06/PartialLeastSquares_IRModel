# PartialLeastSquares_IRModel
Goal: Build A simple interactive dashboard to allow user to import IR spectra from the instrument, train the model as desired and do their own prediction. The dashboard will allow the user to test the effect applying different Savitzky-Golay hyperparameter and see its effect in the final prediction model

[Web App URL](https://share.streamlit.io/nguyenvu06/partialleastsquares_irmodel/main/pls_visualization.py)

### Quick Instruction to test dashboard:

1. Download and unzip the **IR_Data_files.zip**  to used for dashboard testing
2. Included is the **Training_Spectra_X** file. This is the X absorbency features to be used to training
3. The **Wave_number_data** is used to label what is wavelength where the absorbance came from.
4. The y train file is labeled **Expected_conc_for_Training_Y**
5. Follow the instructions on the dashboard to perform training. Once happy with the result, there are 2 test spectra set to test your prediction. 
6. Once a model is trained, and tested to produce happy results, the model can be pickled and saved to your own machine to be loaded for future uses.
7. BONUS STEP: The pickled model can be loaded into [This dashboard](https://nguyenvu06-pls-test-app-pls-test-deployment-kdws9o.streamlitapp.com/) for a continous real time prediction set up. Real time prediction is made off of reading CSV file in a directory of the local computer. 

Feel free to use these files as templates for your own application!



