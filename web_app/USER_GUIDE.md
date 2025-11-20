# ML Portfolio - User Guide

## 🎯 Testing Models with Your Own Data

The ML Portfolio platform allows you to upload your own images, text, or audio files and get real-time predictions from trained machine learning models.

---

## 📸 Image Classification - Upload Your Own Images

### How to Use

1. **Navigate to the Image Classification Model**
   - Go to http://localhost (after starting the app)
   - Click on the "Image Classification" card on the home page

2. **Upload Your Image**

   **Option 1: Drag and Drop**
   - Drag an image file from your computer
   - Drop it into the upload area

   **Option 2: Click to Upload**
   - Click on the upload area
   - Select an image file from your file browser

3. **Supported Image Formats**
   - `.jpg` / `.jpeg`
   - `.png`
   - `.gif`
   - `.bmp`
   - `.webp`
   - Maximum file size: 10MB

4. **Get Predictions**
   - After uploading, click "Run Prediction"
   - Wait for processing (usually < 1 second)
   - View results with confidence scores

### What You'll See

The model will predict one of these 10 classes:
- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

**Results Include:**
- **Top Prediction** - Most likely class with confidence %
- **Top 5 Predictions** - All likely classes ranked by confidence
- **Confidence Bars** - Visual representation of certainty
- **Confidence Scores** - Exact percentages for each class

### Example Usage

```
1. Upload an image of your dog
2. Click "Run Prediction"
3. Results:
   🥇 Dog - 95.2%
   🥈 Cat - 3.1%
   🥉 Deer - 1.2%
   ...
```

### Tips for Best Results

**✅ DO:**
- Use clear, well-lit photos
- Center the subject in the frame
- Use images with single main subjects
- Try different angles of the same object
- Experiment with various image types

**❌ AVOID:**
- Extremely blurry images
- Very dark or overexposed photos
- Images with multiple different subjects
- Non-photographic images (drawings, text, etc.)

**📝 Note:** The model was trained on CIFAR-10 dataset (32x32 pixel images), so it works best with simple, clear images of the 10 classes listed above. Other objects may still get predictions but might not be accurate.

---

## 📝 Text Classification (Coming Soon)

Once the Text Classification model is trained, you'll be able to:

### How to Use

1. Navigate to "Text Classification" model
2. Enter or paste your text in the text box
3. Click "Run Prediction"
4. View sentiment analysis results

### Example Use Cases
- Movie review sentiment (positive/negative)
- Customer feedback analysis
- Social media post sentiment
- Product review classification

---

## 🎤 Audio Processing (Coming Soon)

### Speech Emotion Recognition

1. Upload audio file (.wav, .mp3, .flac)
2. Get emotion predictions:
   - Happy
   - Sad
   - Angry
   - Neutral
   - Fear
   - Disgust
   - Surprise

### Automatic Speech Recognition

1. Upload speech audio file
2. Get transcribed text
3. View confidence scores

---

## 🔍 Understanding Results

### Confidence Scores

- **90-100%** - Very confident prediction
- **70-90%** - Confident prediction
- **50-70%** - Moderate confidence
- **< 50%** - Low confidence (model uncertain)

### Top 5 Predictions

The model shows its top 5 most likely predictions. This helps you understand:
- What else the model considered
- How certain the model is
- Similar categories the model might confuse

### All Probabilities

You can see the raw probability for all classes. This shows:
- Complete distribution of predictions
- Which classes the model ruled out
- Model's overall understanding

---

## 🎨 Model Testing Workflow

### Single Image Testing

```
1. Home Page
   ↓
2. Select Model (e.g., Image Classification)
   ↓
3. Upload File
   ↓
4. Click "Run Prediction"
   ↓
5. View Results
   ↓
6. Upload Another or Clear
```

### Batch Testing (API)

For testing multiple images at once, use the API:

```bash
curl -X POST \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  http://localhost:8000/api/models/image_classification/batch
```

---

## 📊 Viewing Model Performance

### Metrics Dashboard

1. Click "Metrics" in the navigation
2. View performance statistics:
   - Accuracy on test data
   - Training time
   - Model parameters
   - F1 scores

3. Compare models:
   - Side-by-side performance
   - Best performing models
   - Fastest training times

### Training History

For each model, you can see:
- Training and validation loss curves
- Accuracy progression over epochs
- Learning rate schedules
- Confusion matrices

---

## 🔧 Advanced Features

### API Usage

#### Get Model Information
```bash
curl http://localhost:8000/api/models/image_classification
```

#### Make a Prediction
```bash
curl -X POST \
  -F "file=@your_image.jpg" \
  http://localhost:8000/api/models/image_classification/predict
```

#### Get Model Metrics
```bash
curl http://localhost:8000/api/metrics/image_classification/metrics
```

### Response Format

```json
{
  "status": "success",
  "model": "image_classification",
  "result": {
    "prediction": "dog",
    "confidence": 0.9523,
    "class_id": 5,
    "top5_predictions": [
      {
        "class": "dog",
        "confidence": 0.9523,
        "class_id": 5
      },
      {
        "class": "cat",
        "confidence": 0.0312,
        "class_id": 3
      }
    ],
    "all_probabilities": {
      "airplane": 0.0012,
      "automobile": 0.0008,
      ...
    }
  },
  "processing_time_ms": 23.4
}
```

---

## 💡 Example Scenarios

### Scenario 1: Testing a Pet Photo

**Task:** Classify a photo of your pet

1. Take or find a clear photo of your pet
2. Go to Image Classification model
3. Upload the photo
4. Run prediction
5. Check if it correctly identifies dog/cat
6. Try different angles or lighting

**Expected Result:**
- If it's a dog or cat, should have high confidence (>80%)
- May confuse similar-looking animals
- Confidence drops with unclear photos

### Scenario 2: Testing Vehicle Images

**Task:** Test the model on different vehicles

1. Find images of vehicles (cars, trucks, airplanes, ships)
2. Upload each one
3. Compare predictions
4. Note which ones work best

**Expected Result:**
- Clear, centered vehicles → High accuracy
- Side angles work better than front/back
- Modern vs vintage may affect results

### Scenario 3: Testing Edge Cases

**Task:** See how the model handles unusual inputs

1. Try uploading images NOT in the 10 classes
2. Try mixed subjects (dog + cat in one image)
3. Try artistic or cartoon versions
4. Try very zoomed in/out images

**Expected Result:**
- Model will still make predictions
- Confidence will be lower
- May default to most similar class
- Multiple subjects → unpredictable

---

## 🚨 Troubleshooting

### "Failed to load model"
- **Issue:** Model file not found
- **Solution:** Train the model first using the Jupyter notebook
- **Path:** `01_Image_Classification/models/best_model.pt`

### "File too large"
- **Issue:** Image exceeds 10MB
- **Solution:** Compress or resize image before uploading

### "Prediction failed"
- **Issue:** Image format not supported or corrupted
- **Solution:**
  - Convert to .jpg or .png
  - Try a different image
  - Check file isn't corrupted

### "Unexpected prediction"
- **Issue:** Model predicted wrong class
- **Solution:** This is normal! Consider:
  - Model was trained on CIFAR-10 (32x32 images)
  - Some objects are harder to classify
  - Try clearer images or different angles
  - Check if object is in the 10 classes

### Low confidence (<50%)
- **Issue:** Model is uncertain
- **Reasons:**
  - Object not in training classes
  - Unclear/blurry image
  - Multiple subjects
  - Unusual angle or lighting
- **Solution:** Try a different image or accept uncertainty

---

## 📱 Using on Different Devices

### Desktop/Laptop
- Full functionality
- Best experience
- Fast processing
- Large file support

### Tablet
- Fully responsive
- Touch-friendly upload
- All features available
- May be slower for large files

### Mobile
- Works on mobile browsers
- Can use camera to take photos
- Upload from photo library
- Smaller screen but fully functional

---

## 🎓 Educational Use

### Learning About ML Models

**Questions to Explore:**
1. What makes the model confident vs uncertain?
2. Which classes does the model confuse?
3. How does image quality affect predictions?
4. What happens with objects outside the training set?

### Experiments to Try

1. **Same Object, Different Conditions**
   - Same dog, different lighting
   - Same car, different angles
   - Compare confidence scores

2. **Similar Objects**
   - Upload cat and dog images
   - Note confusion patterns
   - Understand decision boundaries

3. **Edge Cases**
   - Toy versions vs real objects
   - Drawings vs photographs
   - Black & white vs color

---

## 🔗 Resources

### API Documentation
- Interactive API docs: http://localhost:8000/api/docs
- ReDoc format: http://localhost:8000/api/redoc

### Model Information
- View on Model page
- Check Metrics Dashboard
- See Training Notebooks

### Source Code
- GitHub repository links on each model page
- Jupyter notebooks for training code
- Model implementation details

---

## 📞 Getting Help

### Common Questions

**Q: Can I upload multiple images at once?**
A: Use the batch API endpoint (see Advanced Features above)

**Q: Can I download prediction results?**
A: Currently view-only, export feature coming soon

**Q: How long does prediction take?**
A: Usually < 1 second for single images

**Q: Can I use this commercially?**
A: This is an educational portfolio project

**Q: Which model should I use?**
A: Depends on your data type:
- Images → Image Classification
- Text → Text Classification (when available)
- Audio → Speech models (when available)

---

## 🎉 Have Fun Testing!

The platform is designed to be:
- **Interactive** - See results instantly
- **Educational** - Learn how ML models work
- **Accessible** - Easy to use for everyone
- **Transparent** - See confidence scores and alternatives

Try uploading different images and explore how the model performs. Understanding both successes and failures helps you learn how machine learning works!

---

**Need more models?** Check the [PROJECT_STATUS.md](../PROJECT_STATUS.md) to see which models are available and coming soon.

**Want to train your own?** Check the Jupyter notebooks in the `notebooks/` directory.
