#!/bin/bash
set -e

# Configuration
PROJECT_ID="plantstory"
REGION="us-central1"
REPOSITORY="cloud-run-apps"
JOB_NAME="customer-service-case-analysis"
IMAGE_NAME="us-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${JOB_NAME}"

echo "=================================="
echo "Deploying Case Processor to Cloud Run Jobs"
echo "=================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPOSITORY}"
echo "Job Name: ${JOB_NAME}"
echo "Image: ${IMAGE_NAME}"
echo "Platform: linux/amd64 (Cloud Run compatible)"
echo "=================================="
echo ""

# Step 0: Check and create secrets if needed
echo "🔐 Checking Secret Manager..."

# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com --project=${PROJECT_ID} --quiet

# Check if secrets exist, create if not
OPENAI_SECRET="openai-sheng-customer-service-case-analysis"
BIGQUERY_SECRET="bigquery-sheng-customer-service-ai"

# Check OpenAI secret
if gcloud secrets describe ${OPENAI_SECRET} --project=${PROJECT_ID} &>/dev/null; then
    echo "  ✅ Secret ${OPENAI_SECRET} exists"
else
    echo "  ➕ Creating secret: ${OPENAI_SECRET}"
    if [ -z "${OPENAI_API_KEY}" ]; then
        echo "  ❌ Error: OPENAI_API_KEY environment variable required to create secret"
        echo "     Please run: export OPENAI_API_KEY=your-key"
        exit 1
    fi
    echo -n "${OPENAI_API_KEY}" | gcloud secrets create ${OPENAI_SECRET} --data-file=- --project=${PROJECT_ID}
fi

# Check BigQuery secret
if gcloud secrets describe ${BIGQUERY_SECRET} --project=${PROJECT_ID} &>/dev/null; then
    echo "  ✅ Secret ${BIGQUERY_SECRET} exists"
else
    echo "  ➕ Creating secret: ${BIGQUERY_SECRET}"
    if [ -z "${BIGQUERY_CREDENTIALS_JSON}" ]; then
        echo "  ❌ Error: BIGQUERY_CREDENTIALS_JSON environment variable required to create secret"
        echo "     Please run: export BIGQUERY_CREDENTIALS_JSON='your-json'"
        exit 1
    fi
    echo -n "${BIGQUERY_CREDENTIALS_JSON}" | gcloud secrets create ${BIGQUERY_SECRET} --data-file=- --project=${PROJECT_ID}
fi

echo "✅ Secrets ready"
echo ""

# Step 1: Setup Artifact Registry
echo "🔧 Setting up Artifact Registry..."
# Configure Docker authentication for Artifact Registry
gcloud auth configure-docker us-docker.pkg.dev --quiet

# Create repository if it doesn't exist
if ! gcloud artifacts repositories describe ${REPOSITORY} --location=us --project=${PROJECT_ID} &>/dev/null; then
    echo "📦 Creating Artifact Registry repository: ${REPOSITORY}"
    gcloud artifacts repositories create ${REPOSITORY} \
      --repository-format=docker \
      --location=us \
      --project=${PROJECT_ID} \
      --description="Docker images for Cloud Run applications"
else
    echo "✅ Repository ${REPOSITORY} already exists"
fi

# Step 2: Build Docker image
echo "🔨 Building Docker image for linux/amd64..."
docker build --platform linux/amd64 -t ${IMAGE_NAME} .

# Step 3: Push to Artifact Registry
echo "📤 Pushing image to Artifact Registry..."
docker push ${IMAGE_NAME}

# Step 4: Deploy to Cloud Run Jobs (using secrets from Secret Manager)
echo "🚀 Deploying to Cloud Run Jobs..."
gcloud run jobs deploy ${JOB_NAME} \
  --image ${IMAGE_NAME} \
  --region ${REGION} \
  --project ${PROJECT_ID} \
  --max-retries 0 \
  --task-timeout 84600 \
  --parallelism 1 \
  --tasks 1 \
  --set-secrets "OPENAI_API_KEY=openai-sheng-customer-service-case-analysis:latest,BIGQUERY_CREDENTIALS_JSON=bigquery-sheng-customer-service-ai:latest" \
  --memory 4Gi \
  --cpu 2

echo ""
echo "=================================="
echo "✅ Deployment Complete!"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Parallelism: 1 (ensures serial execution)"
echo "  Tasks: 1"
echo "  Memory: 4Gi"
echo "  CPU: 2"
echo "  Timeout: 72000s (20 hours)"
echo ""
echo "Test the job manually:"
echo "  gcloud run jobs execute ${JOB_NAME} --region ${REGION} --project ${PROJECT_ID}"
echo ""
echo "View job logs:"
echo "  gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=${JOB_NAME}\" --limit 50 --format json"
echo ""
echo "Note: parallelism=1 ensures only one instance runs at a time."
echo "      If triggered while running, new execution will wait in queue."
echo ""
echo "Next step: Run ./setup_scheduler.sh to configure daily scheduling"
echo "=================================="
