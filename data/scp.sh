FOLDER=/dss/dsshome1/01/ge47wer3/build/src/strong_scaling/1759615720894.031/output_soft/24ranks
# FOLDER=/dss/dsshome1/01/ge47wer3/build/src/lambda_grow/
CURR_TIME=$(date +"%Y%m%d_%H%M%S")/

# scp -r ge47wer3@cool.hpc.lrz.de:$FOLDER $CURR_TIME

rsync -avz --progress ge47wer3@cool.hpc.lrz.de:$FOLDER $CURR_TIME