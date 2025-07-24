#!/bin/bash
echo "Showing logs for all services... (Press Ctrl+C to exit)"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
docker-compose logs -f --tail=100