# makePredictions-functionApp

This function app is activated via a timer trigger that triggers every 15 minutes. It does the following:

## List all squares that have a usable model: list_folders()

The model for each square is stored in individual folders with the UUID of the squares as name of the folder. So naturally, our first step is to retrieve those folder names

## For each folder we do the following:

### Retrieve latest data for that square: get_merged_df()

In each folder, the relevant data is stored in the "merged.csv" file. We retrieve that file and download it to a local pandas frame.

### Retrieve the model for that square: get_model()

For each square that have a folder, it is store as a pickle file called "model.pkl", we download it

### get the latest data: get_latest()
Return the latest relevant data for each the square.
The issue we have been encountering is that there are many gaps in our data, if the gaps are too big, the data is unusable.
To tackle this issue, we do the following:

- take the last 10 rows of the dataframe
- count the number of Nan values in each column and average them over all the columns
- if there are less than 75% Nan values, we repeat the process but look at 10 more rows
- if there are more than 75% Nan values, we return the rows we just analyzed as the latest data   

This helps us to get the relevant latest data without encountering too many Nan values

### get the latest data (temporary): temp_get_latest()
Since we do not have the application constantly running, there is no recent data stored in the storage and the previous function would just return an empty dataframe.   
Instead, we just take the data from 8:50 to 17:20 on 7th December 2023 (a time interval where we have enough data for each square) and return that to the model.   
Keep in mind this is a TEMPORARY method that will be removed once all the part of the application are deployed

### Make prediction for the square: make_prediction()

We predict the next 360 values but we keep only 3 (in 1 hour, 2 hours and 1 day).   

Right now we predict many values but don't use many of them because that would be too many values sent to the business app. In the future, we might be able to scale up and send more of those predicted values

### Send predictions to the business app: send_predictions_to_BA()

We send all the predictions we just made on the new_prediction_queue