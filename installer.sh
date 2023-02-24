# build container
docker build containerfold -t containerfold:blank

# check if the model weights exist in the container directory
if [ ! -d containerfold/params ]; then
    echo "Model weights not found in local directory, please mount the directory and try again"
    exit 1
fi

# move the model weights to the container
echo "Moving model weights to container"
docker run -v $(pwd)/containerfold/params:/params containerfold:blank sh -c "cp -r /params/* /colabfold_batch/ && chown -R root:root /colabfold_batch"

# change the tag
echo "Changing tag"
docker commit $(docker ps -lq) containerfold:latest

# test the container
echo "Testing container"
docker run containerfold:latest colabfold_batch --help
