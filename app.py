from flask import Flask, render_template, request, redirect, url_for
import os

from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import numpy as np
from keras import backend as K
from tensorflow.keras.preprocessing.image import load_img



app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)

    Returns:
      weighted_loss (function): weighted loss function

    """
    # Convert pos_weights and neg_weights to float32
    pos_weights = pos_weights.astype(np.float32)
    neg_weights = neg_weights.astype(np.float32)

# Ensure y_true and y_pred are float32
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value.

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        # initialize loss to zero
        loss = 0.0

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class
            loss += - pos_weights[i] * K.mean(y_true[:,i] * K.log(y_pred[:,i] + epsilon)) \
            - neg_weights[i] * K.mean((1-y_true[:,i]) * K.log(1-y_pred[:,i] + epsilon)) #complete this line
        return loss

        ### END CODE HERE ###
    return weighted_loss


with custom_object_scope({'weighted_loss': get_weighted_loss}):
    model = load_model('mamodel.h5')

IMAGE_DIR = "uploads/"
train_df = pd.read_csv("output.csv")
labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation','No Finding']
layer_name = "conv5_block16_concat"

def get_mean_std_per_batch(df, H=320, W=320):
    sample_data = []
    for img in df["Image"].values:
        image_path = os.path.join("img/", img)
        sample_data.append(
            np.array(load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data, axis=(0, 1, 2, 3))
    std = np.std(sample_data, axis=(0, 1, 2, 3), ddof=1)
    return mean, std


def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    mean, std = get_mean_std_per_batch(df, H=H, W=W)
    x = load_img(img, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

def compute_gradcam(predictions, img, image_dir, df, labels,save_path,
                    layer_name='conv5_block16_concat'):

    pred_df = pd.DataFrame(predictions, columns=labels)

    # Get the top 3 predicted labels
    top_labels = pred_df.iloc[0].nlargest(3).index
    top_values = pred_df.iloc[0].nlargest(3)

    # Display the original image
    percentage = "{:.2%}".format(top_values[0])
    plt.title(f"{top_labels[0]}: sure ratio={percentage}")
    plt.axis('off')
    plt.figure(figsize=(8, 8))
    original_image = load_image(img, image_dir, df, preprocess=False)
    plt.imshow(original_image, cmap='gray')
    plt.savefig('static/result.png')
    plt.close()

def plot_graph(predictions, labels, width=8, height=16):
    # Create a figure with a specific size
    plt.figure(figsize=(width, height))
    
    pred_df = pd.DataFrame(predictions, columns=labels)
    pred_df.loc[0, :].plot.bar()
    
    plt.title("Predictions")
    
    # Save the plot
    plt.savefig('static/predictions.png')

# Your existing code for model loading and image processing functions here...

@app.route('/')
def index():
    return render_template('result.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = 'uploaded_image.png'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('index'))

@app.route('/process', methods=['POST'])
def process_image():
    # Perform your model predictions and processing here using the uploaded image
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
    preprocessed_input = load_image(uploaded_image_path, IMAGE_DIR, train_df)
    predictions = model.predict(preprocessed_input)

    # Generate images and graphs and save them
    compute_gradcam(predictions, uploaded_image_path, IMAGE_DIR, train_df, labels, "result.png")
    plot_graph(predictions, labels)


    pred_d = pd.DataFrame(predictions, columns=labels)

    # Get the top 3 predicted labels
    top_labels = pred_d.iloc[0].nlargest(3).index
    top_values = pred_d.iloc[0].nlargest(3)
    percentage = "{:.2%}".format(top_values[0])

    # Set the result_text, result_image, and result_graph variables
    result_text = f"{top_labels[0]}  :  {percentage}"
    result_image = 'result.png'
    result_graph = 'predictions.png'

    return render_template('result.html', result_text=result_text, result_image=result_image, result_graph=result_graph)

if __name__ == '__main__':
    app.run(debug=True)
