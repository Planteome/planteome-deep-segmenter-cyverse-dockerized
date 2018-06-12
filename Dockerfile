
# +
# build:
#   docker build -t ubisque:uplanteome -f Dockerfile .
# run:
#   docker run --name uplanteome ubisque:uplanteome 
# check:
#   docker logs -t uplanteome
# -

# use ubuntu base image
FROM ubuntu:16.04

# maintenance record
MAINTAINER Phil Daly "pndaly@cyverse.org"

# update the distro
RUN apt-get -y update && apt-get -y upgrade && apt-get -y autoremove

# install software
RUN apt-get -y install python-pip python-setuptools python-dev \
    python-lxml python-tk python-matplotlib python-numpy       \
    python-scipy python-skimage gcc g++ curl

# fix certificate issue?
RUN curl https://bootstrap.pypa.io/get-pip.py | python

# install python dependencies
RUN pip install --upgrade pip setuptools
RUN pip install --ignore-installed scipy==1.0.1
RUN pip install torch==0.4.0 torchvision==0.2.1 numpy==1.14.3 \
    opencv-python==3.4.0.12 scikit-image==0.13.1 lxml==3.7.3  \  
    matplotlib==2.2.2 cython==0.28.2 PyMaxflow==1.2.9         \
    pillow==4.1.1 requests==2.10.0 libtiff==0.4.0             \
    tifffile==0.14.0 bqapi==0.5.9

# copy code
WORKDIR /module/workdir
RUN git clone https://github.com/DimTrigkakis/PlanteomeDeepSegment_0.3.git
RUN mv PlanteomeDeepSegment_0.3 PlanteomeDeepSegment

# run command
ENV PYTHONPATH /module/workdir/PlanteomeDeepSegment:/module/workdir
ENV PATH /module/workdir/PlanteomeDeepSegment:/module/workdir:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CMD [ 'python', 'PlanteomeDeepSegment.py' ]