import sys
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import json
import requests

# Define the mapping from class indices to class names
class_names = [
    "Cassava Bacterial Blight (CBB) : Cassava Bacterial Blight (CBB) is a devastating bacterial disease affecting cassava plants. It is caused by the bacterium Xanthomonas axonopodis pv. manihotis. The disease manifests through symptoms such as angular leaf spots, wilting, chlorosis, necrosis, and eventual plant death. CBB poses a significant threat to cassava production, leading to substantial yield losses and impacting food security in regions where cassava is a staple crop.",
    "Cassava Brown Streak Disease (CBSD) : Cassava Brown Streak Disease (CBSD) is a viral plant disease that primarily affects cassava plants (Manihot esculenta). It is caused by two closely related viruses: Cassava brown streak virus (CBSV) and Ugandan cassava brown streak virus (UCBSV). CBSD is characterized by distinctive brown streaks or necrotic lesions on the stems, petioles, and veins of infected cassava plants.",
    "Cassava Green Mottle (CGM) : The virus is primarily transmitted through vegetative propagation of infected planting material, as well as through the activity of aphid vectors. CGM can lead to significant yield losses, affecting the productivity and quality of cassava crops. In regions where cassava is a staple food, CGM poses a threat to food security and livelihoods.",
    "Cassava Mosaic Disease (CMD) : This disease is characterized by mosaic patterns of light and dark green areas on the leaves, as well as stunted growth and reduced yield. Severe infections can lead to complete crop loss. CMD is a major threat to cassava production in many tropical regions where cassava is a staple food crop.",
    "Healthy"
]

def preprocess_image(image):
    # Normalize [0, 255] to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # Resize the images to 224 x 224
    image = tf.image.resize(image, (224, 224))
    return image

def predict_disease(image_path, model_path):
    image = cv2.imread(image_path)
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    classifier = hub.KerasLayer(model_path)

    probabilities = classifier(image)
    predictions = tf.argmax(probabilities, axis=-1).numpy()  # Convert to NumPy array

    predicted_labels = [class_names[pred] for pred in predictions]
    return predicted_labels

def chat(disease):
    url = "https://chat-gpt26.p.rapidapi.com/"

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": disease
            }
        ]
    }
    headers = {
        "content-type": "application/json",
        "Content-Type": "application/json",
        "X-RapidAPI-Key": "e186862435mshe95204971804cacp171c26jsn386170de0586",
        "X-RapidAPI-Host": "chat-gpt26.p.rapidapi.com"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0 and 'message' in data['choices'][0]:
            output = "Response:" + data['choices'][0]['message']['content']
        else:
            output = "No response content found."
    else:
        output = "Error:" + response.status_code, response.text
    
    return output

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_disease.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = 'models/'
    predictions = predict_disease(image_path, model_path)

    # Prepare the output
    if not predictions:
        output = "Healthy"
    else:
        output = predictions[0] if predictions[0] else "Healthy"

    print(output)  # Print predictions as a plain string
