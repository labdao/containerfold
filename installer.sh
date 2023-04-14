#!/bin/bash
# build container
# docker build containerfold -t containerfold:blank
# docker build protbert -t protbert --no-cache


aws_configure_from_env() {
    # Check if .env file exists
    if [ -f .env ]; then
        # Export the secrets from .env file as environment variables
        export $(grep -v '^#' .env | xargs)

        # Check if required variables are set
        if [ -z "${AWS_ACCESS_KEY_ID}" ] || [ -z "${AWS_SECRET_ACCESS_KEY}" ] || [ -z "${AWS_DEFAULT_REGION}" ]; then
            echo "Required AWS variables are not set in .env file"
            return 1
        fi

        # Configure AWS CLI using environment variables
        aws configure set aws_access_key_id "${AWS_ACCESS_KEY_ID}"
        aws configure set aws_secret_access_key "${AWS_SECRET_ACCESS_KEY}"
        aws configure set default.region "${AWS_DEFAULT_REGION}"

        echo "AWS CLI is now configured"
    else
        echo ".env file not found"
        return 1
    fi
}

# Call the function
aws_configure_from_env

# Add other functions and commands below


# check if the model weights exist in the container directory
#if [ ! -d containerfold/params ]; then
#    echo "Model weights not found in local directory, please mount the directory and try again"
#    exit 1
#fi

# moving weights to the container
#echo "Moving model weights to container"
#docker run -v $(pwd)/containerfold/params:/params containerfold:blank sh -c "cp -r /params/* /colabfold_batch/colabfold/params && chown -R root:root /colabfold_batch"

# changing tags
#echo "Changing tag"
#docker commit $(docker ps -lq) containerfold:latest
#docker tag containerfold:latest public.ecr.aws/p7l9w5o7/containerfold:latest

# testing container
#echo "Testing container"
#docker run containerfold:latest colabfold_batch --help

# pushing container
#docker push public.ecr.aws/p7l9w5o7/containerfold:latest