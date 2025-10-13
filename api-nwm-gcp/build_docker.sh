# Docker build script
#!/bin/bash

# Docker build script for UTH Prediction API
# Usage: ./build_docker.sh [--no-cache] [--tag custom_tag]

set -e  # Exit on any error

# Default values
TAG="uth-prediction:latest"
CACHE_FLAG=""
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            CACHE_FLAG="--no-cache"
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--no-cache] [--tag custom_tag]"
            echo "  --no-cache    Build without cache"
            echo "  --tag         Custom tag for the image (default: uth-prediction:latest)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🐳 Building Docker image for UTH Prediction API"
echo "================================================"
echo "Tag: $TAG"
echo "Build Date: $BUILD_DATE"
echo "VCS Ref: $VCS_REF"
echo "Cache: $([[ -n "$CACHE_FLAG" ]] && echo "disabled" || echo "enabled")"
echo "================================================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check for required files
REQUIRED_FILES=(
    "Dockerfile"
    "requirements.txt"
    "with_saphir_config.py"
    "without_saphir_config.py"
    "README.md"
)

echo "📋 Checking required files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Required file not found: $file"
        exit 1
    fi
    echo "✅ $file"
done

# Build the Docker image
echo "🔨 Building Docker image..."
docker build \
    $CACHE_FLAG \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg VCS_REF="$VCS_REF" \
    --build-arg VERSION="1.0.0" \
    --tag "$TAG" \
    .

# Check if build was successful
if [[ $? -eq 0 ]]; then
    echo "✅ Docker image built successfully!"
    echo "Image size: $(docker images --format "table {{.Size}}" "$TAG" | tail -n 1)"
    
    echo ""
    echo "🚀 Quick Start:"
    echo "# Run with GPU support:"
    echo "docker run -it --gpus all \\"
    echo "  -v /path/to/your/data:/data \\"
    echo "  -v /path/to/your/output:/output \\"
    echo "  $TAG"
    echo ""
    echo "# Run without GPU:"
    echo "docker run -it \\"
    echo "  -v /path/to/your/data:/data \\"
    echo "  -v /path/to/your/output:/output \\"
    echo "  $TAG"
    echo ""
    echo "📚 See README.md for detailed usage instructions"
else
    echo "❌ Docker build failed!"
    exit 1
fi