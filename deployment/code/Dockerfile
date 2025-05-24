FROM ubuntu:24.04
## Install Packages
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    python3-numpy \
    python3-pandas \
    python3-shapely \
    python3-pyproj \
    && rm -rf /var/lib/apt/lists/*

# Create a /work directory within the container, copy everything from the
# build directory and switch there.
RUN mkdir /work
COPY . /work
WORKDIR /work

# test and train scripts should be executable within the container.
RUN chmod +x test.sh
RUN chmod +x train.sh

