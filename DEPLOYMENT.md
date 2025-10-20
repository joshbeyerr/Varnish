# Vercel Deployment Guide for Varnish

This guide will help you deploy your FastAPI + Next.js application to Vercel.

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **GitHub Account**: Your code should be in a GitHub repository
3. **Vercel CLI** (optional): `npm i -g vercel`

## Project Structure

Your project is now configured for Vercel deployment with:

```
├── vercel.json              # Vercel configuration
├── requirements.txt         # Python dependencies for Vercel
├── api/
│   ├── main.py             # FastAPI app (updated for serverless)
│   ├── cloak.py            # Your image processing module
│   └── STW/                # Additional modules
└── frontend/
    ├── package.json        # Next.js dependencies
    └── app/
        └── page.tsx        # Updated frontend
```

## Deployment Steps

### Method 1: Deploy via Vercel Dashboard (Recommended)

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect it's a Next.js project

3. **Configure Environment Variables**:
   - In Vercel dashboard, go to your project settings
   - Add environment variable:
     - `NEXT_PUBLIC_API_URL` = `https://your-app-name.vercel.app/api`

4. **Deploy**:
   - Click "Deploy" and wait for the build to complete

### Method 2: Deploy via Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   vercel
   ```

4. **Set Environment Variables**:
   ```bash
   vercel env add NEXT_PUBLIC_API_URL
   # Enter: https://your-app-name.vercel.app/api
   ```

## Important Notes

### Python Dependencies
- The `requirements.txt` in the root includes all necessary Python packages
- Vercel will automatically install these for the serverless functions
- Some packages like PyTorch might be large - Vercel has size limits

### Image Processing Limitations
- Vercel serverless functions have a 50MB memory limit
- Processing time is limited to 30 seconds (configured in vercel.json)
- Large images might fail to process due to memory constraints

### File Storage
- Vercel serverless functions are stateless
- Files are processed in temporary directories
- Images are returned as base64 data instead of file downloads

## Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check that all Python dependencies are in `requirements.txt`
   - Ensure your Python code doesn't use file system operations that persist

2. **Memory Issues**:
   - Reduce image processing complexity
   - Consider resizing images before processing
   - Use lighter versions of ML models

3. **Timeout Issues**:
   - Increase timeout in `vercel.json` (max 30s for hobby plan)
   - Optimize your image processing code

4. **CORS Issues**:
   - The API is configured to allow all origins
   - Make sure your frontend URL matches the deployed URL

### Alternative Deployment Options

If Vercel doesn't work well for your image processing needs:

1. **Railway**: Better for Python apps with file storage
2. **Render**: Good for full-stack applications
3. **Google Cloud Run**: More flexible for ML workloads
4. **AWS Lambda**: For serverless with more resources

## Testing Your Deployment

1. **Check API Health**:
   - Visit `https://your-app.vercel.app/api/health`
   - Should return: `{"status": "healthy", "message": "Varnish API is running"}`

2. **Test Image Upload**:
   - Go to your deployed frontend
   - Try uploading a small image
   - Check if processing works

3. **Monitor Logs**:
   - Use Vercel dashboard to check function logs
   - Look for any errors during image processing

## Next Steps

1. **Optimize for Production**:
   - Add proper error handling
   - Implement rate limiting
   - Add authentication if needed

2. **Scale Up**:
   - Consider upgrading Vercel plan for more resources
   - Or migrate to a more suitable platform for ML workloads

3. **Monitor Performance**:
   - Set up monitoring and alerts
   - Track processing times and success rates
