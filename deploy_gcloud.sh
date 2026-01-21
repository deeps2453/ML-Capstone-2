#!/bin/bash

# Deploy to Google Cloud Run
# This script builds and deploys the salary prediction service to Google Cloud Run

set -e


echo "DEPLOYING TO GOOGLE CLOUD RUN"

# Configuration
PROJECT_ID="your-gcp-project-id"  # Change this
SERVICE_NAME="salary-prediction"
REGION="us-central1"
MEMORY="2Gi"
CPU="2"
MAX_INSTANCES="10"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo ""
echo "[1/4] Setting up GCP project..."
gcloud config set project ${PROJECT_ID}

echo ""
echo "[2/4] Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

echo ""
echo "[3/4] Building and deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --source . \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory ${MEMORY} \
    --cpu ${CPU} \
    --max-instances ${MAX_INSTANCES} \
    --set-env-vars PYTHONUNBUFFERED=1

echo ""
echo "[4/4] Getting service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo ""
echo "DEPLOYMENT SUCCESSFUL!"
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test the service:"
echo "  curl ${SERVICE_URL}/health"
echo ""
echo "  curl -X POST ${SERVICE_URL}/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"Is_top_5_League\": 1, \"Based_rich_nation\": 1, ...}'"
echo ""
