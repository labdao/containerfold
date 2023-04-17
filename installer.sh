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

# Function to list the available repositories in the public ECR
list_ecr_repositories() {
    ecr_repositories=$(aws ecr-public describe-repositories --query 'repositories[].repositoryName' --output text)
    echo "$ecr_repositories"
}

# Function to list the local Docker images based on the Dockerfiles
list_local_repositories() {
    dockerfiles_with_label=$(grep -r "LABEL container=" . --include="Dockerfile" | awk -F'[=:]' '{print $3}')
    echo "$dockerfiles_with_label"
}

# Function to manage the ECR repositories
manage_ecr_repositories() {
    # Get the list of ECR repositories and local images
    ecr_repositories=$(list_ecr_repositories)
    echo "ECR Public repositories: $ecr_repositories"
    dockerfiles_with_label=$(list_local_repositories)
    echo "Dockerfiles found: $dockerfiles_with_label"

    # Create a new repository called XYZ if the repository does not yet exist in the public ECR
    for label in $dockerfiles_with_label; do
        repository_name=$(echo "$label" | tr -d '[:space:]"') # Remove whitespace and quotes
        if [[ ! $ecr_repositories =~ $repository_name ]]; then
            echo "Creating new ECR Public repository: $repository_name"
            aws ecr-public create-repository --repository-name "$repository_name"
        else
            echo "Repository $repository_name already exists"
        fi
    done
}


build_and_push_container() {
    local container_label="$1"

    # 1) Build the container with the name and target -t XYZ:latest
    echo "Building container: $container_label:latest"
    docker build "$container_label" -t "$container_label:latest" --no-cache

    # 2) Check the ECR registry identity of the XYZ docker container
    registry_uri=$(aws ecr-public describe-repositories --repository-names "$container_label" --query 'repositories[0].repositoryUri' --output text)
    echo "ECR registry URI: $registry_uri"

    # 3) Retag the XYZ:latest container to the ECR identity
    echo "Retagging $container_label:latest to $registry_uri:latest"
    docker tag "$container_label:latest" "$registry_uri:latest"

    # 4) Push the docker container
    echo "Pushing container to ECR: $registry_uri:latest"
    docker push "$registry_uri:latest"
}


# Call the function
aws_configure_from_env
login_ecr
manage_ecr_repositories

# Call the function with the container label as a CLI argument
if [ $# -eq 1 ]; then
    build_and_push_container "$1"
else
    echo "Usage: $0 <container-label>"
    exit 1
fi