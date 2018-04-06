#!/bin/sh
if [ -f files.zip ]; then
    echo "already downloaded"
    exit 1
fi
wget -O files.zip https://s3.amazonaws.com/drivendata/data/4/public/da1dd36a-a497-42c7-b3f3-4a225944bdba.zip && unzip files.zip -d data_in
