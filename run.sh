#!/usr/bin/bash
./build-cv4.sh
cd cv4_output
#./yolact -m ../yolact_uint8.tmfile -i yolact.jpg -r 1 -t 1

./yolact -m ../yolact_50_uint8.tmfile -i dog.jpg -r 1 -t 1