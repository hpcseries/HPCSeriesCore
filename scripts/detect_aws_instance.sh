#!/bin/bash
# detect_aws_instance.sh - AWS EC2 Instance Metadata Detection
#
# Queries AWS EC2 Instance Metadata Service v2 (IMDSv2) to retrieve instance information.
# Gracefully handles non-EC2 environments.
#
# Usage:
#   ./detect_aws_instance.sh
#
# Output:
#   Comma-separated key=value pairs (e.g., "instance-id=i-1234,instance-type=c7i.xlarge,...")
#   Or "not-ec2=true" if not running on EC2
#
# Exit codes:
#   0 - Success (EC2 or non-EC2)
#   1 - Error querying metadata
#

set -euo pipefail

# AWS IMDSv2 endpoint
METADATA_ENDPOINT="http://169.254.169.254"
TOKEN_TTL=21600  # 6 hours

# Timeout settings (short timeouts to fail fast on non-EC2)
CONNECT_TIMEOUT=1
MAX_TIME=2

# Step 1: Get IMDSv2 token
token=$(curl -s -X PUT "${METADATA_ENDPOINT}/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: ${TOKEN_TTL}" \
    --connect-timeout "$CONNECT_TIMEOUT" \
    --max-time "$MAX_TIME" \
    2>/dev/null || echo "")

# Check if token retrieval failed (not on EC2)
if [ -z "$token" ]; then
    echo "not-ec2=true"
    exit 0
fi

# Step 2: Query metadata with token
get_metadata() {
    local path="$1"
    curl -s -H "X-aws-ec2-metadata-token: $token" \
        "${METADATA_ENDPOINT}/latest/meta-data/${path}" \
        --connect-timeout "$CONNECT_TIMEOUT" \
        --max-time "$MAX_TIME" \
        2>/dev/null || echo ""
}

# Collect instance metadata
instance_id=$(get_metadata "instance-id")
instance_type=$(get_metadata "instance-type")
availability_zone=$(get_metadata "placement/availability-zone")
region=$(echo "$availability_zone" | sed 's/[a-z]$//')

# Extract instance family from instance type (e.g., c7i.xlarge → c7i)
if [ -n "$instance_type" ]; then
    instance_family=$(echo "$instance_type" | cut -d. -f1)
else
    instance_family=""
fi

# Verify we got at least instance-id (confirms EC2)
if [ -z "$instance_id" ]; then
    echo "not-ec2=true"
    exit 0
fi

# Output comma-separated metadata
echo "instance-id=${instance_id},instance-type=${instance_type},instance-family=${instance_family},az=${availability_zone},region=${region}"
