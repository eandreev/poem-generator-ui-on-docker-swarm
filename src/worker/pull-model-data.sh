if [ ! -e /var/model-data ]; then
    echo "==================================="
    echo "Downloading model data..."
    echo
    mkdir -p /var/model-data
    cd /var/model-data
    aws s3 cp --recursive s3://poem-generator-prod /var/model-data/
    ls -lh /var/model-data
    echo "DONE"
    echo "==================================="
fi