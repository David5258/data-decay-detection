# AWS Deployment Info

## Infrastructure
- EC2 Instance: t3.micro (us-east-1)
- Public IP: 98.92.51.8
- Security Group: sg-0f4717e19a2049f53

## Services
- S3 Bucket: data-decay-detection-models
- ECR Repository: 567764214258.dkr.ecr.us-east-1.amazonaws.com/data-decay-detection

## API Endpoints
- Docs:    http://98.92.51.8/docs
- Health:  http://98.92.51.8/health
- Detect:  http://98.92.51.8/detect (POST)

## Important
- Remember to stop EC2 instance when not using it to avoid charges
- Run: aws ec2 stop-instances --instance-ids <instance-id> --region us-east-1
