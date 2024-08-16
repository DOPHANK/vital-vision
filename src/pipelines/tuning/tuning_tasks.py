# src/ocr/tuning_tasks.py

from celery import Celery
from src.pipelines.tuning.tune_hyperparameters import run_tuning

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def scheduled_tuning():
    run_tuning()

# In a separate terminal, start the Celery worker:
# celery -A src.ocr.tuning_tasks worker --loglevel=info
