#!/usr/bin/env python3
"""
Test AWS local access keys configuration
Run this on Jetson to verify everything works
"""
import yaml
import boto3
from datetime import datetime

def test_aws_local_keys():
    print("🧪 Testing AWS Local Access Keys Configuration")
    print("=" * 50)
    
    try:
        # Load your config
        with open("aws_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Extract credentials
        aws_key = config.get("aws_access_key_id", "NOT FOUND")
        aws_secret = config.get("aws_secret_access_key", "NOT FOUND")
        region = config.get("aws_region", "NOT FOUND")
        
        print(f"📋 Configuration Loaded:")
        print(f"   AWS Key: {aws_key[:10]}...{aws_key[-4:]}")
        print(f"   Region: {region}")
        print(f"   S3 Bucket: {config.get('s3_bucket', 'NOT FOUND')}")
        
        # Create session
        session = boto3.Session(
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=region
        )
        
        # Test S3
        s3 = session.client('s3')
        response = s3.list_buckets()
        print(f"\n✅ AWS Connection Successful!")
        print(f"   Account: {response['Owner']['DisplayName']}")
        print(f"   Buckets: {len(response['Buckets'])} found")
        
        # Test specific bucket
        target_bucket = config.get('s3_bucket')
        bucket_names = [b['Name'] for b in response['Buckets']]
        
        if target_bucket in bucket_names:
            print(f"✅ Target bucket '{target_bucket}' exists")
        else:
            print(f"⚠ Target bucket '{target_bucket}' NOT found")
            print(f"   Available: {', '.join(bucket_names[:3])}...")
        
        # Test Textract
        textract = session.client('textract')
        print(f"✅ Textract client created")
        
        print("\n🎉 All AWS services accessible with local keys!")
        print("\n📝 Next steps:")
        print("   1. Run: python detect.py --config config.yaml --aws-config aws_config.yaml")
        print("   2. Check S3 bucket for uploaded images")
        print("   3. Monitor CloudWatch for logs")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Verify aws_config.yaml format")
        print("   2. Check IAM permissions")
        print("   3. Ensure internet connectivity on Jetson")
        print("   4. Validate AWS credentials in IAM Console")

if __name__ == "__main__":
    test_aws_local_keys()