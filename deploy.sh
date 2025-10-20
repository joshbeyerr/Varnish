#!/bin/bash

# Varnish Vercel Deployment Script
echo "🚀 Preparing Varnish for Vercel deployment..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit for Vercel deployment"
fi

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📥 Installing Vercel CLI..."
    npm install -g vercel
fi

echo "🔐 Please login to Vercel..."
vercel login

echo "🚀 Deploying to Vercel..."
vercel

echo "✅ Deployment complete!"
echo "📝 Don't forget to:"
echo "   1. Set NEXT_PUBLIC_API_URL environment variable in Vercel dashboard"
echo "   2. Test your deployment"
echo "   3. Check the deployment guide in DEPLOYMENT.md"
