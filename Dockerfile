FROM python:3.6.4

WORKDIR /fashion

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBUG 0

# copy project
COPY ./api/ /fashion/api/
COPY ./training/ /fashion/training/
COPY ./requirements.txt /fashion/

# Install requirements
RUN pip install --upgrade pip
RUN pip install -r /fashion/requirements.txt

# install psycopg2
RUN apt-get update \
    && apt-get install -y gcc python3-dev musl-dev \
    && apt-get install -y postgresql \
    && pip install psycopg2

# add and run as non-root user
RUN adduser --disabled-password myuser
USER myuser

# run gunicorn
CMD cd api/mysite && gunicorn mysite.wsgi --bind 0.0.0.0:$PORT

