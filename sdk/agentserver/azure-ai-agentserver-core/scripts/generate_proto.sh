#!/usr/bin/env bash
# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# Regenerate Python gRPC stubs from the proto definition.
#
# Prerequisites:
#   pip install grpcio-tools
#
# Usage:
#   cd sdk/agentserver/azure-ai-agentserver-core
#   bash scripts/generate_proto.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROTO_SRC="$PROJECT_DIR/protos"
PROTO_OUT="$PROJECT_DIR/azure/ai/agentserver/core/proto"

echo "Generating proto stubs..."
echo "  Source:  $PROTO_SRC/agent_service.proto"
echo "  Output:  $PROTO_OUT/"

python -m grpc_tools.protoc \
  -I "$PROTO_SRC" \
  --python_out="$PROTO_OUT" \
  --grpc_python_out="$PROTO_OUT" \
  "$PROTO_SRC/agent_service.proto"

# Fix the bare import in the generated _grpc file to use a relative import.
# protoc generates `import agent_service_pb2 as ...` which doesn't work
# inside a Python package.
sed -i 's/^import agent_service_pb2 as/from . import agent_service_pb2 as/' \
  "$PROTO_OUT/agent_service_pb2_grpc.py"

echo "Done. Generated files:"
ls -la "$PROTO_OUT"/agent_service_pb2*.py
