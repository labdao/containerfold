# build container
docker build containerfold -t containerfold:blank
docker build protbert -t protbert --no-cache

# check if the model weights exist in the container directory
if [ ! -d containerfold/params ]; then
    echo "Model weights not found in local directory, please mount the directory and try again"
    exit 1
fi

# moving weights to the container
echo "Moving model weights to container"
docker run -v $(pwd)/containerfold/params:/params containerfold:blank sh -c "cp -r /params/* /colabfold_batch/colabfold/params && chown -R root:root /colabfold_batch"

# changing tags
echo "Changing tag"
docker commit $(docker ps -lq) containerfold:latest
docker tag containerfold:latest public.ecr.aws/p7l9w5o7/containerfold:latest

# testing container
echo "Testing container"
docker run containerfold:latest colabfold_batch --help

# pushing container
docker push public.ecr.aws/p7l9w5o7/containerfold:latest