"""
Module to register mlflow model
"""
import mlflow
def register_model():
    run = mlflow.active_run()
    run_id = run.info.run_id
    model_url = "runs:/" + run_id + "/house_price_model"
    mlflow.register_model(model_url, "house_price_model-reg")
    return None