# Use a lightweight Python image for the runtime
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

RUN pip install uv

COPY requirements.txt ./

RUN uv pip install --system -r requirements.txt

# Copy application code into the container
COPY . /app

# Expose the port the app will run on
EXPOSE 10000

# Use uvicorn to run the FastAPI app defined in app.py
# Erase your old CMD line and paste this instead
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
