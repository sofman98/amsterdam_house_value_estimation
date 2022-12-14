# Start from the official Python base image.
FROM python:3.9

# Set the current working directory to /code.
WORKDIR /code

# Copy the file with the requirements to the /code directory.
COPY ./requirements.txt /code/requirements.txt

# Install the package dependencies in the requirements file.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the current directory inside the /code directory.
COPY . /code

# Run the training and evaluation of the model
RUN ["python", "-m", "src.scripts.train_and_evaluate_model"]

# Set the command to run the uvicorn server.
CMD ["uvicorn", "src.app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
