# Use a slim Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the dependencies, including Jupyter for running the notebook
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyter

# Copy the rest of the project files into the container
COPY . .

# Expose the port Jupyter will run on
EXPOSE 8888

# Command to run when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]