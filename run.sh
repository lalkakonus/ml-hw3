#!/bin/bash

# Data download
DIR="data"
mkdir -p $DIR

for (( i = 0; i < 3; i++ ))
do
    read -p "Download data? [y/n]:  " answer
    if [[ "$answer" == "y" ]] || [[ "$answer" == "n" ]]; then
        break
    fi
done

if [ "$answer" == "y" ]; then
	echo "Loading data from Google Drive..."

	FILEID="1_wi0JFIQADsjhjtbIoYnNUkAqhi_BypW"
	FILENAME=$DIR"/data.tar.gz"
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- |
	sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILEID -O $FILENAME &&
	rm -rf /tmp/cookies.txt
	echo "Loading data from Google Drive...OK"

	echo "Unpacking data archive..."
	tar -zxvf $FILENAME
	echo "Unpacking data archive...OK"

	echo "Remove archive file..."
	rm --verbose $FILENAME
	echo "Remove archive file...OK"


	echo "Data loading finished"
elif [ "$answer" != "n" ]; then
    echo "3 time incorrect input"
    exit 1
fi

# Run python ranking code
echo "Run python code..."
python3 "src/main.py"
