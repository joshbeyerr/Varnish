@echo off
echo 🚀 Preparing Varnish for Vercel deployment...

REM Check if git is initialized
if not exist ".git" (
    echo 📦 Initializing git repository...
    git init
    git add .
    git commit -m "Initial commit for Vercel deployment"
)

REM Check if vercel CLI is installed
where vercel >nul 2>nul
if %errorlevel% neq 0 (
    echo 📥 Installing Vercel CLI...
    npm install -g vercel
)

echo 🔐 Please login to Vercel...
vercel login

echo 🚀 Deploying to Vercel...
vercel

echo ✅ Deployment complete!
echo 📝 Don't forget to:
echo    1. Set NEXT_PUBLIC_API_URL environment variable in Vercel dashboard
echo    2. Test your deployment
echo    3. Check the deployment guide in DEPLOYMENT.md

pause
