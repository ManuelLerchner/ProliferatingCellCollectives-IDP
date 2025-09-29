FOLDER=/dss/dsshome1/01/ge47wer3/build/src/vtk_output_hard
CURR_TIME=$(date +"%Y%m%d_%H%M%S")/

# scp -r ge47wer3@cool.hpc.lrz.de:$FOLDER $CURR_TIME

rsync -avz --progress ge47wer3@cool.hpc.lrz.de:$FOLDER $CURR_TIME