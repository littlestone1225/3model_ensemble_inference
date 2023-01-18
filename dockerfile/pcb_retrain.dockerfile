# FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

ARG USER=adev
ARG UID=1000
ARG GID=1000
ARG AOI_DIR_NAME="AOI_PCB_Inference"

ENV DISPLAY :11
ENV DEBIAN_FRONTEND noninteractive


RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update && \
    apt-get install -y sudo vim git wget curl zip unzip dmidecode && \
    apt-get install -y net-tools iputils-ping apt-utils && \
    # For opencv-python
    # (pkg-config --cflags opencv)
    # (pkg-config --modversion opencv)
    apt-get install -y libsm6 libxrender-dev libopencv-dev && \
    # Install make
    apt-get install -y build-essential

# Install python2.7 and python3.6
RUN apt-get install -y python python-dev && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    #apt-get install -y python2.7 python-dev && \
    apt-get install -y python3.6 python3-pip python3.6-dev && \
    cd /usr/bin && \
    rm python3 && \
    ln -s python3.6 python3 && \
    rm python && \
    ln -s python3.6 python

#RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o /root/get-pip2.py && \
#    python2 /root/get-pip2.py && \
#    rm /root/get-pip2.py && \
    
RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o /root/get-pip.py && \
    python /root/get-pip.py && \
    rm /root/get-pip.py && \
    curl https://bootstrap.pypa.io/pip/3.6/get-pip.py -o /root/get-pip3.py && \
    python3 /root/get-pip3.py && \
    rm /root/get-pip3.py

# Install python2.7 package

# Install python3.6 package
RUN apt install unixodbc-dev --yes && \
    pip3 install afs2-datasource afs2-model && \
    pip3 install opencv-python==4.1.1.26 && \
    pip3 install pyinstaller==3.6 && \
    pip3 install nuitka==0.6.8.4 && \
    pip3 install flask && \
    pip3 install psutil && \
    pip3 install filelock==3.0.12 && \
    pip3 install SharedArray && \
    pip3 install cloudpickle && \
    pip3 install omegaconf

# RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html && \
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install termcolor==1.1.0 && \
    pip3 install protobuf==3.13.0 && \
    pip3 install Cython==0.29.21 && \
    pip3 install matplotlib && \
    pip3 install pycocotools==2.0 && \
    pip3 install scipy==1.5.4 && \
    pip3 install PyYAML==3.11

# Set the home directory to our user's home.
ENV USER=$USER
ENV HOME="/home/$USER"
ENV AOI_DIR_NAME=$AOI_DIR_NAME

RUN echo "Create $USER account" &&\
    # Create the home directory for the new $USER
    mkdir -p $HOME &&\
    # Create an $USER so our program doesn't run as root.
    groupadd -r -g $GID $USER &&\
    useradd -r -g $USER -G sudo -u $UID -d $HOME -s /sbin/nologin -c "Docker image user" $USER &&\
    # Set root user no password
    mkdir -p /etc/sudoers.d &&\
    echo "$USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER && \
    chmod 0440 /etc/sudoers.d/$USER && \
    # Chown all the files to the $USER
    chown -R $USER:$USER $HOME

# Change to the $USER
WORKDIR $HOME
USER $USER
