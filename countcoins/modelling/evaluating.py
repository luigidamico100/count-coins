from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def get_all_predictions(model, data_path, df_annotations, only_test_set=False):

    predictions_list = []

    if only_test_set:
        df_annotations = df_annotations.query("set=='test'")

    for filename, image_series in tqdm(df_annotations.iterrows()):
        currency = image_series.currencies
        image_path = os.path.join(data_path, 'coins_images', currency, filename)
        # image = read_image(image_path=image_path, plot_image=False)
        y_pred = model.predict(image_path=image_path)

        prediction_info = {
            'currency': image_series.currencies,
            'set': image_series.set,
            'y_true': image_series.coins_count,
            'y_pred': y_pred,
        }
        predictions_list.append(prediction_info)

    df_predictions = pd.DataFrame(predictions_list)

    return df_predictions


def compute_metrics(df_predictions):
    metrics_list = []

    for set_ in df_predictions['set'].unique():
        df_predictions_set = df_predictions.query("set==@set_")
        y_true = df_predictions_set['y_true']
        y_pred = df_predictions_set['y_pred']

        metrics_list += [{
                'currency': 'all',
                'set': set_,
                'value': metric_fun(y_true, y_pred),
                'metric': metric_fun.__name__
            } for metric_fun in (mean_absolute_error, root_mean_squared_error)]
        

        for currency in df_predictions['currency'].unique():
            df_predictions_currency_set = df_predictions_set.query("currency==@currency")
            y_true = df_predictions_currency_set['y_true']
            y_pred = df_predictions_currency_set['y_pred']
            metrics_list += [{
                    'currency': currency,
                    'set': set_,
                    'value': metric_fun(y_true, y_pred),
                    'metric': metric_fun.__name__
                } for metric_fun in (mean_absolute_error, root_mean_squared_error)]
            
    df_metrics = pd.DataFrame(metrics_list)

    return df_metrics
