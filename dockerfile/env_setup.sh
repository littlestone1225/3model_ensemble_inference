#!/bin/bash

# Color
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'

IFS=$'\n'
function Fun_EvalCmd()
{
    cmd_list=$1
    i=0
    for cmd in ${cmd_list[*]}
    do
        ((i+=1))
        printf "${GREEN}${cmd}${NC}\n"
        eval $cmd
    done
}

lCmdList=(
            "cd $HOME/$AOI_DIR_NAME" \
            "git clone -b aoi_pcb https://github.com/littlestone1225/fvcore.git" \
            "git clone -b aoi_pcb https://github.com/littlestone1225/CenterNet2.git detectron2" \
            "git clone -b aoi_pcb https://github.com/littlestone1225/darknet.git"
         )
Fun_EvalCmd "${lCmdList[*]}"
echo ""

##################################
# *****  Build detectron2  ***** #
##################################
# git clone -b advan_pcb https://github.com/tkyen1110/fvcore.git
# git clone -b ctr2_pcb https://github.com/tkyen1110/CenterNet2.git detectron2

fvcore_status=`echo $(pip list | grep fvcore) | awk -F' ' '{print $1}'`
if [ "$fvcore_status" != "fvcore" ]
then
    lCmdList=(
                "cd $HOME/$AOI_DIR_NAME" \
                "pip install --user -e fvcore"
             )
    Fun_EvalCmd "${lCmdList[*]}"
else
    echo -e "${GREEN}fvcore already exists${NC}"
fi
echo ""

detectron2_status=`echo $(pip list | grep detectron2) | awk -F' ' '{print $1}'`
if [ "$detectron2_status" != "detectron2" ]
then
    lCmdList=(
                "cd $HOME/$AOI_DIR_NAME" \
                "pip install --user -e detectron2"
             )
    Fun_EvalCmd "${lCmdList[*]}"
else
    echo -e "${GREEN}detectron2 already exists${NC}"
fi
echo ""

##################################
# ******   Build darknet  ****** #
##################################
# git clone -b advan_pcb https://github.com/tkyen1110/darknet.git
darknet_status=`ls $HOME/$AOI_DIR_NAME/darknet | grep libdarknet.so`
if [ -z "$darknet_status" ]
then
    lCmdList=(
                "cd $HOME/$AOI_DIR_NAME/darknet" \
                "make clean" \
                "make"
             )
    Fun_EvalCmd "${lCmdList[*]}"
else
    echo -e "${GREEN}darknet already exists${NC}"
fi
echo ""

##################################
# ******   Run inference  ****** #
##################################
lCmdList=(
            "cd $HOME/$AOI_DIR_NAME/config" \
            "python config.py" \
            "cd $HOME/$AOI_DIR_NAME/inference" \
            "python inference.py"
         )
Fun_EvalCmd "${lCmdList[*]}"

/bin/bash
