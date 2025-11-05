#!/usr/bin/env python3
"""
Production Training Launcher
Real market data only - no simulations
"""

import os
import sys
import subprocess
import platform

def print_header():
    print("=" * 70)
    print("  PRODUCTION TRAINING - Real Market Data Only")
    print("=" * 70)
    print()
    print("This will train the RL agent on REAL data from:")
    print("  - Polygon S3 (Massive) for US Stocks")
    print("  - Binance for Crypto")
    print("  - Yahoo Finance for supplementary data + news sentiment")
    print()
    print("NO SIMULATIONS - Only actual market outcomes")
    print()
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print()

def install_dependencies():
    print("[1/2] Installing dependencies...")
    print("-" * 70)
    
    if os.path.exists("requirements_production.txt"):
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements_production.txt"],
                check=True
            )
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print("⚠️  requirements_production.txt not found")
        return False
    
    print()
    return True

def start_training():
    print("[2/2] Starting production training...")
    print("-" * 70)
    print()
    
    try:
        subprocess.run(
            [sys.executable, "production_train.py"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return False
    
    return True

def main():
    print_header()
    
    try:
        input("Press Enter to continue...")
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    
    print()
    
    if not install_dependencies():
        sys.exit(1)
    
    if start_training():
        print()
        print("=" * 70)
        print("✅ Training complete!")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("❌ Training failed or was interrupted")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
