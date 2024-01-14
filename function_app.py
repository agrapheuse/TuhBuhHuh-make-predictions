import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from io import StringIO
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
import pika
import json
import uuid
from enum import Enum
import h5py
from io import BytesIO
from keras.models import load_model

app = func.FunctionApp()

@app.function_name("makePrediction")
@app.schedule(schedule="0 */15 * * * *", arg_name="myTimer", run_on_startup=True,
              use_monitor=False) 
def makePrediction(myTimer: func.TimerRequest) -> None:
    logging.info('Python timer trigger function executed.')

    # Connect to our blob storage
    connection_string = "DefaultEndpointsProtocol=https;AccountName=datalaketuhbehhuh;AccountKey=C2te9RgBRHhIH8u3tydAsn9wNd4umdD2axq1ZdcfKh7CZRpL04+D4H6QinE/gckMTUA/dFj1kFpd+ASt4+/8ZA==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    folders = list_folders(blob_service_client, "model")

    for folder in folders:
        # Get the merged.csv file for the folder (contains all the data for the square)
        merged_df = get_merged_df(blob_service_client, folder)

        # Get the model for the square
        model = get_model(blob_service_client, folder)

        # latest_data_df = get_latest(merged_df)
        latest_data_df = temp_get_latest(merged_df)

        # If there is no recent data, skip the square
        if latest_data_df is None:
            logging.info(f"Folder {folder} has no recent data.")
            continue

        last_timestamp = latest_data_df['timestamp'].iloc[-1]
        latest_data_df = latest_data_df.drop(columns='timestamp')

        logging.info(f"Making prediction for square with UUID: {folder}")
        predictions = make_prediction(model, latest_data_df, last_timestamp)
        predictions["squareUUID"] = folder.split('/')[0]

        logging.info(f"Sending predictions to business app")
        send_predictions_to_BA(predictions)


def list_folders(blob_service_client, container_name):
    # Connect to the container and extract a list of all the blobs
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs()

    folder_set = set()

    # Extract the folder names from the blob names and add them to a set of all the folder names
    for blob in blob_list:
        blob_name = blob.name
        folder_name = "/".join(blob_name.split('/')[:-1])
        folder_set.add(folder_name)

    return folder_set

# Download the merged.csv file for the square into a DataFrame
def get_merged_df(blob_service_client, folder):
    df = download_blob_to_file(blob_service_client, "csv", f"history/{folder}/merged.csv")
    return df

def download_blob_to_file(blob_service_client, container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob()

    csv_file = StringIO(blob_data.readall().decode('utf-8'))
    df = pd.read_csv(csv_file)
    return df

def get_latest(df):
    # Drop columns with no data
    columns_to_drop = []
    for i in df.columns:
        if df[i].count() == 0:
            columns_to_drop.append(i)

    df = df.drop(columns=columns_to_drop)

    # goes through the tail of the datafrale 10 rows at a time and checks if the average
    # number of non-null values is less than 75% of the number of rows. if it, that
    # means that there is not enough data for that square if we take more data so we return the dataframe
    df_without_timestamp = df.drop(columns='timestamp')
    for i in range(10, df.shape[0], 10):
        temp_df = df_without_timestamp.iloc[-i:]
        count_avg = temp_df.count().mean()
        if count_avg < i * 0.75:
            return temp_df

def temp_get_latest(df):
    logging.warning(f"Getting latest data")
    # Takes the data from 8:50 to 17:20 on 7th December 2023 (a time interval where we
    # have enough data for each square)
    start_timestamp = '2023-12-07 08:50:00'
    end_timestamp = '2023-12-07 17:20:00'
    filtered_df = df[(df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)]

    threshold = 0.5  # Set the threshold percentage
    nan_percentage = filtered_df.isna().mean()
    columns_with_high_nan = nan_percentage[nan_percentage > threshold].index.tolist()
    # Drop the column if it has more than 50% NaN values
    filtered_df = filtered_df.drop(columns=columns_with_high_nan)
    return filtered_df

def get_model(blob_service_client, folder):
    blob_client = blob_service_client.get_blob_client(container="model", blob=f"{folder}/model.h5")

    logging.warning(f"Attempting to download blob for {folder}/model.h5")
    downloader = blob_client.download_blob(0)
    logging.warning(f"Blob downloaded successfully")

    with BytesIO() as f:
        downloader.readinto(f)
        with h5py.File(f, 'r') as h5file:
            logging.warning(f"Attempting to load model from h5 file")
            model = load_model(h5file)
            model.summary()
            logging.warning(f"Model loaded successfully from Blob Storage for folder: {folder}")

    return model

def download_model_to_file(blob_service_client, container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob()
    return blob_data.readall()

def make_prediction(model, latest_data_df, last_timestamp):
    # Interpolate the missing values
    latest_data_df = latest_data_df.interpolate(method='linear')

    # Normalize the new data using the same scaler as during training
    scaler = StandardScaler()  # Assuming you used StandardScaler during training
    new_data_scaled = scaler.fit_transform(latest_data_df)

    # Create sequences for prediction
    window_size = 36  # Same as during training
    X_new = []

    for i in range(len(new_data_scaled) - window_size):
        X_new.append(new_data_scaled[i:i + window_size])

    X_new = np.array(X_new)

    # Make predictions for the next 10 values
    predictions_actual = []

    # Update X_new for the next prediction
    for i in range(10):
        prediction = model.predict(X_new[:, -window_size:, :])  # Predict using the last window_size values
        predictions_actual.append(prediction)
        prediction = prediction.reshape(prediction.shape[0], 1, prediction.shape[1])
        # Update X_new for the next prediction
        X_new = np.concatenate([X_new, prediction], axis=1)

    # Reshape predictions_actual and concatenate them with new_data
    predictions_actual = np.array(predictions_actual).reshape(-1, latest_data_df.shape[1])
    predicted_data = pd.DataFrame(predictions_actual, columns=latest_data_df.columns)

    # Denormalize the predicted data
    predicted_data_denormalized = scaler.inverse_transform(predicted_data)

    # Round the predicted data to one decimal place
    predicted_data_denormalized_rounded = np.round(predicted_data_denormalized, 1)
    prediction_df = pd.DataFrame(predicted_data_denormalized_rounded, columns=latest_data_df.columns).head(288)
    prediction_df = prediction_df.applymap(lambda x: max(0, x))
    prediction_df = prediction_df.round(1)

    last_timestamp = pd.to_datetime(last_timestamp) + pd.to_timedelta(5, unit='m')
    dti = pd.date_range(last_timestamp, periods=288, freq="5T")
    prediction_df['timestamp'] = dti
    prediction_df = prediction_df[['timestamp'] + [col for col in prediction_df.columns if col != 'timestamp']]
    indices_to_select = [11, 35, -1]  # List of indices to select

    predictions_final_df = prediction_df.iloc[indices_to_select]
    try:
        prediction_df[prediction_df['HUMIDITY'] > 100] = 100
    except KeyError:
        pass

    time_made = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    predictions_final_df["timeMade"] = time_made

    columns_to_drop = ['PEDESTRIAN', 'BIKE', 'V85', 'HEAVY']
    columns_to_drop_existing = [col for col in columns_to_drop if col in predictions_final_df.columns]

    if columns_to_drop_existing:
        predictions_final_df = predictions_final_df.drop(columns=columns_to_drop_existing)
    return predictions_final_df

# event header class
class EventCatalog(Enum):
    NEW_PREDICTION_DATA = 'NEW_PREDICTION_DATA'

class EventHeader:
    def __init__(self, eventID, eventCatalog):
        self.eventID = eventID
        self.eventCatalog = eventCatalog


def send_predictions_to_BA(predictions):
    df_melted = pd.melt(predictions, id_vars=['timestamp', 'timeMade', 'squareUUID'], var_name='valueType', value_name='value')
    # connect to rabbitmq using the right credentials
    connection_params = pika.ConnectionParameters(
        host='goose.rmq2.cloudamqp.com',
        port=5672,
        virtual_host='ljbtjfzn',
        credentials=pika.PlainCredentials('ljbtjfzn', 'v6hsm9rB5nI8FQnMQxRZUug081s_zPA3')
    )

    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()

    # declare the queue
    queue_name = 'new_prediction_queue'
    channel.queue_declare(queue=queue_name, durable=True)
        
    # convert the dataframe to dict
    df_melted['timestamp'] = df_melted['timestamp'].astype(str)
    data_list = df_melted.to_dict(orient='records')

    # create the event message
    event_header = EventHeader(uuid.uuid4(), EventCatalog.NEW_PREDICTION_DATA)
    event_message_dict = {
        "eventHeader": {
            "eventID": str(event_header.eventID),
            "eventCatalog": event_header.eventCatalog.value
        },
        "eventBody": f"{json.dumps(data_list)}"
    }
    # convert the event message to json
    json_payload = json.dumps(event_message_dict)

    # send the event message to the queue
    channel.basic_publish(exchange='', routing_key=queue_name, body=json_payload)
    logging.info(f"sent new predictions to queue {queue_name}")

    # close the connection
    connection.close()