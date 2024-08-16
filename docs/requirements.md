
# Explanation of Packages in requirements.txt

## 1. fastapi==0.95.0
   - **Purpose**: FastAPI is a modern, fast (high-performance) web framework for building APIs with Python. It is used to create the RESTful API for the OCR-to-LLM system, handling incoming requests, processing, and responding with results.
   - **Usage**: It handles HTTP requests for uploading images, invoking OCR and LLM processing, and returning the results to the client.

## 2. uvicorn==0.21.1
   - **Purpose**: Uvicorn is a lightning-fast ASGI server implementation, using `uvloop` and `httptools`. It is designed to serve FastAPI applications in a production environment.
   - **Usage**: Uvicorn runs the FastAPI application, making it accessible over HTTP.

## 3. gunicorn==20.1.0
   - **Purpose**: Gunicorn is a WSGI HTTP server for raunning Python web applications. It is commonly used in conjunction with Uvicorn to manage multiple worker processes.
   - **Usage**: In a production environment, Gunicorn can manage multiple Uvicorn worker processes to serve the FastAPI application efficiently.

## 4. easyocr==1.5.0
   - **Purpose**: EasyOCR is a deep learning-based OCR tool that supports multiple languages. It is used to extract text from images or scanned documents.
   - **Usage**: It processes uploaded images to extract the text, which is then passed to the LLM for further processing.

## 5. paddleocr==2.5.0
   - **Purpose**: PaddleOCR is another OCR tool based on PaddlePaddle, designed for multilingual text recognition. It offers additional flexibility and accuracy for complex OCR tasks.
   - **Usage**: It can be used as an alternative or complement to EasyOCR, especially for languages that require more advanced processing.

## 6. transformers==4.30.2
   - **Purpose**: The Transformers library by Hugging Face provides APIs and tools to work with state-of-the-art pre-trained models like BERT, GPT, LLAMA, etc.
   - **Usage**: It is used to load and interact with the LLAMA 3.1 language model, enabling text processing and generation tasks.

## 7. celery==5.2.7
   - **Purpose**: Celery is a distributed task queue that enables asynchronous task processing. It helps offload time-consuming tasks like OCR and LLM processing to background workers.
   - **Usage**: Celery is used to handle OCR and LLM tasks asynchronously, ensuring that the FastAPI application remains responsive to incoming requests.

## 8. redis==4.6.0
   - **Purpose**: Redis is an in-memory data structure store that serves as a message broker for Celery. It queues tasks and helps manage background processes.
   - **Usage**: Redis is used in conjunction with Celery to manage the task queue, ensuring tasks are distributed to workers efficiently.

## 9. psycopg2-binary==2.9.6
   - **Purpose**: Psycopg2 is a PostgreSQL adapter for Python. It allows Python applications to interact with PostgreSQL databases.
   - **Usage**: It is used to connect the application to the PostgreSQL database, where data such as processed text, logs, and user feedback are stored.

## 10. sqlalchemy==1.4.41
   - **Purpose**: SQLAlchemy is a powerful ORM (Object-Relational Mapping) library that allows developers to interact with databases using Python objects instead of writing raw SQL queries.
   - **Usage**: SQLAlchemy is used to define database models, manage database sessions, and perform CRUD operations on the PostgreSQL database.

## 11. docker==6.0.1
   - **Purpose**: Docker is a platform that allows you to automate the deployment of applications inside lightweight, portable containers.
   - **Usage**: Docker is used to containerize the entire application, making it easier to deploy and manage on different environments.

## 12. docker-compose==1.29.2
   - **Purpose**: Docker Compose is a tool for defining and running multi-container Docker applications. It allows you to manage multiple services (like the web app, database, and Redis) together.
   - **Usage**: Docker Compose is used to orchestrate the application stack, ensuring all services start and run correctly.

## 13. pytest==7.3.1
   - **Purpose**: Pytest is a testing framework that makes it easy to write simple and scalable test cases for Python code.
   - **Usage**: Pytest is used to write and run unit tests for the OCR service, LLM service, pipeline, and API endpoints, ensuring the application works as expected.

## 14. coverage==7.2.5
   - **Purpose**: Coverage.py measures code coverage during test execution, indicating how much of your code is covered by tests.
   - **Usage**: It is used alongside Pytest to measure and report on the test coverage of the codebase.

## 15. loguru==0.6.0
   - **Purpose**: Loguru is a library that simplifies logging in Python applications, providing an easier-to-use and more powerful alternative to the standard `logging` module.
   - **Usage**: Loguru is used to handle logging throughout the application, capturing and storing logs for debugging, monitoring, and auditing purposes.
