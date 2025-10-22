#!/bin/bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}--- Apex Singularity Nexus (ASN) Startup Script ---${NC}"

# Function to handle errors
handle_error() {
    echo -e "${RED}ERROR: $1${NC}"
    exit 1
}

# 1. Verify UI Directory Exists inside nexus/
if [ ! -d "nexus/ui/dist" ]; then
    handle_error "UI directory not found at nexus/ui/dist."
fi

# 2. Build the Rust Nexus Crate (Core Engine, Python Bindings, and UI Server)
echo -e "\n${YELLOW}[1/4] Building Rust Nexus Crate (Release Mode)...${NC}"
cd nexus || handle_error "Nexus directory not found."

# Standard cargo build
cargo build --release || handle_error "Rust compilation failed."
cd ..
echo -e "${GREEN}[OK] Rust Nexus build complete.${NC}"

# 3. Install Python Dependencies (Optional but recommended)
echo -e "\n${YELLOW}[2/4] Verifying Python Dependencies for Forge...${NC}"
# If using a virtual environment, activate it here (e.g., source venv/bin/activate)
pip install -r forge/requirements.txt > /dev/null 2>&1 || echo -e "${YELLOW}Warning: Failed to install Python requirements. Forge may fail.${NC}"
echo -e "${GREEN}[OK] Python dependencies verified.${NC}"


# 4. Start the Rust Nexus Live Engine
echo -e "\n${YELLOW}[3/4] Starting Rust Nexus Live Engine...${NC}"
# Start the engine. The executable name is 'nexus_tester'.
# We run it in the background so we can open the browser.
./nexus/target/release/nexus_tester &
NEXUS_PID=$!
echo -e "${GREEN}[OK] Nexus Engine starting (PID: $NEXUS_PID).${NC}"

# Give the engine time to initialize the web server
sleep 3

# 5. Open the UI
echo -e "\n${YELLOW}[4/4] Opening ASN Crucible Dashboard...${NC}"
# Detect OS and open the browser automatically
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3030
elif command -v open &> /dev/null; then
    open http://localhost:3030
elif command -v start &> /dev/null; then
    # For Windows (e.g., WSL/Git Bash)
    start http://localhost:3030
else
    echo "Could not automatically open browser. Please navigate to http://localhost:3030"
fi

echo -e "\n${CYAN}--- System Started ---${NC}"
echo "The Nexus Engine is running."
echo "To run the evolution process (Forge), start it separately (e.g., cd forge && python3 gp_framework.py)."
echo "To stop the system, press Ctrl+C in this terminal."

# Trap Ctrl+C to kill the background process cleanly
trap "echo -e '\nStopping Nexus Engine...'; kill $NEXUS_PID; exit" SIGINT

# Keep the script running to monitor the background process
wait $NEXUS_PID