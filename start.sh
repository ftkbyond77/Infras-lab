#!/bin/bash

cleanup() {
    echo "Stopping Service"
    docker stop horse2zebra-container 2>/dev/null
    
    kill $(jobs -p) 2>/dev/null
    echo "Done"

}

trap cleanup EXIT

echo "Building and Starting Backend"

docker build -t horse2zebra-app .

docker run --rm -d -p 8080:8080 --name horse2zebra-container horse2zebra-app

echo "Backend running on port 8080"

echo "Checking Frontend"

if [ ! -d "frontend" ]; then
    echo "Frontend folder not found. Creating Next.js app"

    npx create-next-app@latest frontend
else
    echo "Frontend folder exists. Skipping creation."
fi

echo "Starting Fronend"

cd frontend

npm run dev &

sleep 5

echo "Opening Browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "http://localhost:3000"     # Mac
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "http://localhost:3000" # Linux
elif [[ "$OSTYPE" == "msys" ]]; then
    start "http://localhost:3000"    # Windows (Git Bash)
fi

wait