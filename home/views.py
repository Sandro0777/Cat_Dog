from django.shortcuts import render
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # Import the model loading method
from django.conf import settings
from tensorflow.keras.utils import custom_object_scope
import tensorflow_hub as hub
# Load your trained machine learning model
import os
model_path = os.path.join(settings.BASE_DIR, '3rdsemm.h5')
model = load_model(
       (model_path),
       custom_objects={'KerasLayer':hub.KerasLayer}
)





def preprocess_image(image):
    # Resize the image to 224x224 pixels (adjust as needed)
    resized_image = cv2.resize(image, (224, 224))
    
    # Normalize pixel values to be between 0 and 1
    normalized_image = resized_image / 255.0
    
    # Expand dimensions to match the model's input shape (1, 224, 224, 3)
    input_image = np.expand_dims(normalized_image, axis=0)
    image_reshaped = np.reshape(normalized_image, [1, 224, 224, 3])
    
    return image_reshaped

def classify_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        uploaded_image = request.FILES['image'].read()
        nparr = np.fromstring(uploaded_image, np.uint8)
        input_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess the uploaded image
        input_image = preprocess_image(input_image)

        # Perform prediction using the loaded model
        input_prediction = model.predict(input_image)

        # Set a threshold for prediction confidence
        confidence_threshold = 0.5  # Adjust this threshold as needed

        # Check if the maximum prediction confidence is below the threshold
        if np.max(input_prediction) < confidence_threshold:
            result = 'Unknown Animal'
        else:
            # Check if the prediction confidence for dog (class 1) is greater than cat (class 0)
            if input_prediction[0][1] > input_prediction[0][0]:
                result = 'The image represents a Dog'
            else:
                result = 'The image represents a Cat'
        return render(request, 'result.html', {'result': result})
    else:
        return render(request, 'upload_form.html')
