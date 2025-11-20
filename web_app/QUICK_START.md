# Quick Start Guide - Testing Your Own Images

## 🚀 Get Started in 3 Steps

### 1. Start the Application

```bash
cd web_app
./start.sh
```

Wait for the message: **"✅ Application is running!"**

### 2. Open Your Browser

Go to: **http://localhost**

### 3. Upload and Test!

1. Click on **"Image Classification"** card
2. **Drag & drop** your image OR **click to browse**
3. Click **"Run Prediction"**
4. View results instantly! 🎉

---

## 📸 What Images Can I Upload?

### ✅ Best Results With:
- Dogs 🐕
- Cats 🐱
- Cars/Trucks 🚗
- Airplanes ✈️
- Ships 🚢
- Horses 🐴
- Birds 🐦
- Frogs 🐸
- Deer 🦌

### 📋 Requirements:
- **Format:** JPG, PNG, GIF, BMP
- **Size:** Under 10MB
- **Quality:** Clear, well-lit photos work best

---

## 🎯 Example Workflow

```
1. Take a photo of your dog
   ↓
2. Open http://localhost
   ↓
3. Click "Image Classification"
   ↓
4. Drag your photo to the upload area
   ↓
5. Click "Run Prediction"
   ↓
6. See: "Dog - 95.2% confidence" 🎉
```

---

## 💻 Test via Command Line (Optional)

### Single Image
```bash
cd web_app
python test_upload.py ~/Pictures/my_dog.jpg
```

### Multiple Images
```bash
python test_upload.py --batch image1.jpg image2.jpg image3.jpg
```

### Model Info
```bash
python test_upload.py --info
```

---

## 📊 Understanding Results

### Confidence Levels
- **90-100%** 🟢 Very Confident - Model is very sure
- **70-90%** 🟡 Confident - Likely correct
- **50-70%** 🟠 Moderate - Could be accurate
- **<50%** 🔴 Uncertain - Model not sure

### Top 5 Predictions
You'll see the top 5 most likely predictions:

```
🥇 Dog        ████████████████████ 95.2%
🥈 Cat        ██                   3.1%
🥉 Deer       █                    1.2%
   Horse                           0.3%
   Bird                            0.2%
```

This shows what else the model considered!

---

## 🎨 Try These Examples

### Easy Wins
✅ Clear photo of a single dog
✅ Car on a clean background
✅ Airplane in the sky
✅ Bird on a branch

### Challenging Cases
🤔 Multiple animals in one photo
🤔 Very zoomed in/out
🤔 Unusual angles
🤔 Dark or blurry images

### Fun Experiments
🎮 Upload a toy car vs real car
🎮 Cartoon vs photograph
🎮 Different dog breeds
🎮 Objects NOT in the 10 classes

---

## 🔧 Troubleshooting

### "API is not running"
→ Run `./start.sh` in the web_app directory

### "File too large"
→ Compress image or use smaller file (<10MB)

### "Low confidence prediction"
→ Normal! Try:
- Clearer image
- Better lighting
- Center the subject
- Use simpler backgrounds

### Wrong prediction
→ Also normal! Remember:
- Model trained on 10 specific classes
- Works best on similar objects
- Quality affects accuracy

---

## 📱 Use on Mobile

Yes! Works on phones:
1. Open browser on phone
2. Go to your computer's IP: `http://192.168.x.x`
3. Use camera to take photos
4. Upload and test!

---

## 🎓 Learn More

**Full Guide:** See [USER_GUIDE.md](USER_GUIDE.md)

**API Docs:** http://localhost:8000/api/docs

**Model Metrics:** Click "Metrics" in the app

---

## 💡 Pro Tips

1. **Test the same object multiple times**
   - Different angles
   - Different lighting
   - Compare confidence scores

2. **Try edge cases**
   - What happens with non-animals?
   - How does it handle text?
   - Test with abstract images

3. **Batch testing**
   - Test many images at once via API
   - Compare results across similar images

4. **View detailed metrics**
   - Click "Metrics" in navigation
   - See model performance on test data
   - Compare training times

---

## 🚀 Ready to Test?

```bash
cd web_app
./start.sh
```

Then go to: **http://localhost**

**Have fun exploring!** 🎉

---

## ❓ Need Help?

- **Full Documentation:** [USER_GUIDE.md](USER_GUIDE.md)
- **Deployment Guide:** [DEPLOYMENT.md](DEPLOYMENT.md)
- **Project Status:** [../PROJECT_STATUS.md](../PROJECT_STATUS.md)

---

**Remember:** The model was trained on CIFAR-10 dataset, so it works best with the 10 classes listed above. Feel free to experiment with any images to see how it performs!
