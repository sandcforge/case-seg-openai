#!/bin/bash
set -e

# Configuration
PROJECT_ID="plantstory"
REGION="us-central1"
JOB_NAME="customer-service-case-analysis"
SCHEDULER_NAME="customer-service-case-analysis-daily"
SCHEDULE="0 2 * * *"  # Run at 2:00 AM every day (UTC)
TIMEZONE="America/Los_Angeles"  # Adjust to your timezone

echo "=================================="
echo "Setting up Cloud Scheduler for Daily Execution"
echo "=================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Job Name: ${JOB_NAME}"
echo "Schedule: ${SCHEDULE} (${TIMEZONE})"
echo "  -> Runs at 2:00 AM daily"
echo "=================================="
echo ""

# Check if scheduler already exists
if gcloud scheduler jobs describe ${SCHEDULER_NAME} --location=${REGION} --project=${PROJECT_ID} &>/dev/null; then
    echo "⚠️  Scheduler job '${SCHEDULER_NAME}' already exists."
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Updating existing scheduler..."
        gcloud scheduler jobs update http ${SCHEDULER_NAME} \
          --location=${REGION} \
          --project=${PROJECT_ID} \
          --schedule="${SCHEDULE}" \
          --time-zone="${TIMEZONE}" \
          --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
          --http-method=POST \
          --oauth-service-account-email="${PROJECT_ID}@appspot.gserviceaccount.com"
    else
        echo "❌ Cancelled. Keeping existing scheduler."
        exit 0
    fi
else
    echo "📅 Creating new scheduler job..."
    gcloud scheduler jobs create http ${SCHEDULER_NAME} \
      --location=${REGION} \
      --project=${PROJECT_ID} \
      --schedule="${SCHEDULE}" \
      --time-zone="${TIMEZONE}" \
      --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
      --http-method=POST \
      --oauth-service-account-email="${PROJECT_ID}@appspot.gserviceaccount.com"
fi

echo ""
echo "=================================="
echo "✅ Scheduler Setup Complete!"
echo "=================================="
echo ""
echo "Schedule Details:"
echo "  Name: ${SCHEDULER_NAME}"
echo "  Schedule: ${SCHEDULE}"
echo "  Timezone: ${TIMEZONE}"
echo "  Runs: Every day at 2:00 AM"
echo ""
echo "Useful commands:"
echo "  Test scheduler manually:"
echo "    gcloud scheduler jobs run ${SCHEDULER_NAME} --location=${REGION} --project=${PROJECT_ID}"
echo ""
echo "  View scheduler status:"
echo "    gcloud scheduler jobs describe ${SCHEDULER_NAME} --location=${REGION} --project=${PROJECT_ID}"
echo ""
echo "  Pause scheduler:"
echo "    gcloud scheduler jobs pause ${SCHEDULER_NAME} --location=${REGION} --project=${PROJECT_ID}"
echo ""
echo "  Resume scheduler:"
echo "    gcloud scheduler jobs resume ${SCHEDULER_NAME} --location=${REGION} --project=${PROJECT_ID}"
echo ""
echo "  Delete scheduler:"
echo "    gcloud scheduler jobs delete ${SCHEDULER_NAME} --location=${REGION} --project=${PROJECT_ID}"
echo "=================================="
