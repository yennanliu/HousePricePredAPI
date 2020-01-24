FROM ubuntu:18.04
MAINTAINER "yen"

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PYTHONPATH="$PYTHONPATH:/REST/my_app"
COPY . /my_app
WORKDIR /my_app

RUN apt-get update \ 
    && apt-get install -y vim \
    && conda install --file requirements.txt \
    && pip freeze list 

RUN conda --version

CMD ["python", "/my_app/api/app.py"]

# expose ports
EXPOSE 8000 