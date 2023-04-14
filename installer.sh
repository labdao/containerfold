#!/bin/bash

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

login_ecr() {
    # Log in to ECR Public using AWS CLI
    aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
    if [ $? -eq 0 ]; then
        echo "Successfully logged in to ECR Public"
    else
        echo "Failed to log in to ECR Public"
        return 1
    fi
}

manage_ecr_repositories() {
    # 1) Check the public ECR for the available repositories
    ecr_repositories=$(aws ecr-public describe-repositories --query 'repositories[].repositoryName' --output text)
    echo "ECR Public repositories: $ecr_repositories"

    # 2) Check the available directories for Dockerfiles and list all LABEL container=XYZ
    dockerfiles_with_label=$(grep -r "LABEL container=" . --include="Dockerfile" | awk -F'[=:]' '{print $3}')
    echo "Dockerfiles found: $dockerfiles_with_label"

    # 3) Create a new repository called XYZ if the repository does not yet exist in the public ECR
    for label in $dockerfiles_with_label; do
        repository_name=$(echo "$label" | tr -d '[:space:]') # Remove whitespace
        if [[ ! $ecr_repositories =~ $repository_name ]]; then
            echo "Creating new ECR Public repository: $repository_name"
            aws ecr-public create-repository --repository-name "$repository_name"
        else
            echo "Repository $repository_name already exists"
        fi
    done
}

# Call the function


# Call the function
aws_configure_from_env
login_ecr
manage_ecr_repositories

## TODO - add MoLeR

## TODO remove
# Add other functions and commands below

# build container
# docker build containerfold -t containerfold:blank
# docker build protbert -t protbert --no-cache

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