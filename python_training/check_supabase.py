"""
Quick test to verify Supabase connection from GPU server
"""
import os
import sys
from dotenv import load_dotenv

print("🔍 Testing Supabase connection...\n")

# Load environment variables
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not url:
    print("❌ SUPABASE_URL not found in .env file")
    sys.exit(1)

if not key:
    print("❌ SUPABASE_SERVICE_ROLE_KEY not found in .env file")
    sys.exit(1)

print(f"✅ SUPABASE_URL found: {url}")
print(f"✅ SUPABASE_SERVICE_ROLE_KEY found: {key[:20]}...")

try:
    from supabase import create_client
    print("\n📦 Supabase client imported successfully")
except ImportError:
    print("\n❌ Supabase package not installed")
    print("   Run: pip install supabase")
    sys.exit(1)

try:
    supabase = create_client(url, key)
    print("✅ Supabase client created")
    
    # Test query
    result = supabase.table("training_runs").select("*").limit(1).execute()
    print(f"✅ Query successful! Found {len(result.data)} rows")
    
    # Test insert
    test_data = {
        "run_name": "connection_test",
        "phase": "test",
        "hyperparams": {},
        "status": "testing"
    }
    insert_result = supabase.table("training_runs").insert(test_data).execute()
    print(f"✅ Insert successful! Created run with ID: {insert_result.data[0]['id']}")
    
    # Clean up
    supabase.table("training_runs").delete().eq("run_name", "connection_test").execute()
    print("✅ Cleanup successful")
    
    print("\n🎉 All tests passed! Supabase connection is working correctly.")
    print("   You're ready to start distributed training!")
    
except Exception as e:
    print(f"\n❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check that SUPABASE_SERVICE_ROLE_KEY is correct")
    print("2. Verify internet connection")
    print("3. Try: curl https://rgpgssvakgutmgejazjq.supabase.co")
    sys.exit(1)
