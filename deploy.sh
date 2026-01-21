#!/bin/bash

# Football Player Salary Prediction - Cloud Deployment Script
# This script deploys the model to AWS ECS

set -e


echo "SALARY PREDICTION MODEL - CLOUD DEPLOYMENT"


# Configuration
AWS_REGION="us-east-1"
ECR_REPO_NAME="salary-prediction"
ECS_CLUSTER_NAME="ml-cluster"
ECS_SERVICE_NAME="salary-prediction-service"
TASK_FAMILY="salary-prediction-task"
IMAGE_TAG="latest"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed"
    echo "Please install it: https://aws.amazon.com/cli/"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    echo "Please install it: https://docs.docker.com/get-docker/"
    exit 1
fi

echo ""
echo "[1/7] Getting AWS Account ID..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account ID: $AWS_ACCOUNT_ID"

echo ""
echo "[2/7] Creating ECR Repository (if not exists)..."
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${AWS_REGION} || \
    aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${AWS_REGION}

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"
echo "ECR Repository URI: ${ECR_URI}"

echo ""
echo "[3/7] Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${ECR_URI}

echo ""
echo "[4/7] Building Docker image..."
docker build -t ${ECR_REPO_NAME}:${IMAGE_TAG} .

echo ""
echo "[5/7] Tagging Docker image..."
docker tag ${ECR_REPO_NAME}:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}

echo ""
echo "[6/7] Pushing Docker image to ECR..."
docker push ${ECR_URI}:${IMAGE_TAG}

echo ""
echo "[7/7] Creating/Updating ECS Task Definition..."

# Create task definition JSON
cat > task-definition.json <<EOF
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "salary-prediction",
      "image": "${ECR_URI}:${IMAGE_TAG}",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${TASK_FAMILY}",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

# Register task definition
aws ecs register-task-definition \
    --cli-input-json file://task-definition.json \
    --region ${AWS_REGION}

echo ""
echo "DEPLOYMENT COMPLETE!"
echo ""
echo "Next steps to complete the deployment:"
echo ""
echo "1. Create ECS Cluster (if not exists):"
echo "   aws ecs create-cluster --cluster-name ${ECS_CLUSTER_NAME} --region ${AWS_REGION}"
echo ""
echo "2. Create ECS Service:"
echo "   aws ecs create-service \\"
echo "     --cluster ${ECS_CLUSTER_NAME} \\"
echo "     --service-name ${ECS_SERVICE_NAME} \\"
echo "     --task-definition ${TASK_FAMILY} \\"
echo "     --desired-count 2 \\"
echo "     --launch-type FARGATE \\"
echo "     --network-configuration \"awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}\" \\"
echo "     --region ${AWS_REGION}"
echo ""
echo "3. Create Application Load Balancer and configure it"
echo ""
echo "Or deploy to Cloud Run (Google Cloud):"
echo "   gcloud run deploy salary-prediction \\"
echo "     --image ${ECR_URI}:${IMAGE_TAG} \\"
echo "     --platform managed \\"
echo "     --region us-central1 \\"
echo "     --allow-unauthenticated"
echo ""
echo "================================================"

# Clean up
rm task-definition.json
