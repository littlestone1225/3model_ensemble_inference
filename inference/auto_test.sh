#!/bin/bash
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'

# Absolute path to this script.
SCRIPT=$(readlink -f "$0")

# Absolute path this script is in.
SCRIPT_PATH=$(dirname "$SCRIPT")

# ls image/ | head -n 10 | xargs -i cp image/{} factory_demo
ls_result=`ls $SCRIPT_PATH/factory_img_1`
ls_result=$(echo ${ls_result})
IFS=" " read -a images <<< $ls_result

for (( i=0; i<${#images[@]}; i+=1 ))
do
  if [[ ($i -ge 0 && $i -le 0) ]]
  then
    echo $i, "cp $SCRIPT_PATH/factory_img_1/${images[$i]} $SCRIPT_PATH/factory_img"
    cp $SCRIPT_PATH/factory_img_1/${images[$i]} $SCRIPT_PATH/factory_img
    sleep 2
  fi
done
