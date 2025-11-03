#!/bin/bash
echo "======================================================================"
echo "  SYSTEM VERIFICATION - Run before training"
echo "======================================================================"
echo
echo "This will verify all components are working correctly"
echo
read -p "Press Enter to continue..."

python verify_system.py

echo
echo "======================================================================"
read -p "Press Enter to continue..."
