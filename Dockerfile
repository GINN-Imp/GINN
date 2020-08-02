FROM python:3.7

ENV LANG C.UTF-8

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY . .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory

# command to run on container start
#CMD cd models && python GINN.py --config-file config/config-simple-interval.json


