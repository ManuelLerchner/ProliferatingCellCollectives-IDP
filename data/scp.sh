FOLDER=/dss/dsshome1/01/ge47wer3/build/src/huge/
# FOLDER=/dss/dsshome1/01/ge47wer3/build/src/lambda_grow/
CURR_TIME=$(date +"%Y%m%d_%H%M%S")/

# scp -r ge47wer3@cool.hpc.lrz.de:$FOLDER $CURR_TIME

rsync -avz --progress ge47wer3@cool.hpc.lrz.de:$FOLDER $CURR_TIME