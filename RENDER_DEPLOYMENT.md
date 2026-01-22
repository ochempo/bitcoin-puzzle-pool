# RENDER DEPLOYMENT GUIDE
## FREE Bitcoin Puzzle Pool Deployment

### Prerequisites
- GitHub account (free)
- Render account (free) at https://render.com

---

## STEP 1: Push Code to GitHub

```bash
cd /home/ochempo

# Initialize git (if not already)
git init
git add .
git commit -m "Bitcoin Puzzle Pool deployment"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/bitcoin-puzzle-pool.git
git branch -M main
git push -u origin main
```

---

## STEP 2: Connect to Render

1. Go to **https://render.com**
2. Click **Sign up** (free)
3. Sign in with GitHub
4. Click **New +** → **Web Service**
5. Select your `bitcoin-puzzle-pool` repository
6. Click **Connect**

---

## STEP 3: Configure Deployment

When asked, fill in:

| Field | Value |
|-------|-------|
| **Name** | bitcoin-puzzle-pool |
| **Environment** | Docker |
| **Region** | Oregon (US) |
| **Plan** | Free |
| **Branch** | main |

---

## STEP 4: Deploy

1. Click **Create Web Service**
2. Render will automatically:
   - Build your Docker image
   - Deploy to their servers
   - Give you a live URL

**Your app will be live at:**
```
https://bitcoin-puzzle-pool.onrender.com
```

---

## STEP 5: Test

Once deployed, test the API:

```bash
# Health check
curl https://bitcoin-puzzle-pool.onrender.com/api/v1/health

# Get puzzles
curl https://bitcoin-puzzle-pool.onrender.com/api/v1/puzzles

# Get subscription plans
curl https://bitcoin-puzzle-pool.onrender.com/api/v1/subscriptions/plans
```

---

## KEY BENEFITS

✅ **100% FREE** - No credit card needed
✅ **Auto-deploys** from GitHub (push = deploy)
✅ **Auto-restart** if crashed
✅ **SSL/HTTPS** included
✅ **Easy scaling** when needed

---

## TROUBLESHOOTING

**Deployment stuck?**
- Check Render dashboard logs
- Ensure Dockerfile exists: `/home/ochempo/Dockerfile`
- Verify all dependencies in `requirements.txt`

**App crashes?**
- View logs in Render dashboard
- Check port is 8080 (Render requirement)
- Restart deployment: Dashboard → Manual Deploy

**URL not working?**
- Wait 3-5 minutes for first deploy
- Check Render dashboard status (should be "Live")

