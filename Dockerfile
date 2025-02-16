# FROM python:3.8
# FROM python:3.10
# ADD requirements.txt /
# RUN pip install -r /requirements.txt
# RUN mkdir /connect4_app/
# RUN chmod -R 777 /connect4_app
# ADD PYTHONFILE.py /connect4_app
# ENV PYTHONUNBUFFERED=1
# CMD [ "python", "connect4_app/PYTHONFILE.py" ]


# Use Python 3.10 as the base image
FROM python:3.10
# Copy requirements.txt to the container
ADD requirements.txt /connect4_app/
# Set working directory to /connect4_app
WORKDIR /connect4_app
# Install dependencies
RUN pip install -r requirements.txt
# Copy the entire project folder into the container
ADD . /connect4_app/
# Set permissions for the project folder
RUN chmod -R 777 /connect4_app
# Set environment variable for unbuffered logs
ENV PYTHONUNBUFFERED=1
# Run the main Python file
CMD ["python", "play_game.py"]