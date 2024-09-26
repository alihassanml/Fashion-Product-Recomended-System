# Fashion Product Recommendation System
A Fashion Product Recommendation System built using deep learning and computer vision techniques. This app uses a pre-trained ResNet50 model to extract image features and recommends similar fashion items based on the input image. The project is deployed using Streamlit, providing an interactive interface for users to find similar fashion products easily.

## Features
- Upload an image to find visually similar fashion products.
- Displays information about the recommended items, including gender, article type, and season.
- Utilizes a pre-trained ResNet50 model for feature extraction.
- Uses Nearest Neighbors algorithm for similarity search.

## Demo
Check out the live demo of the Kaggle: [Fashion Product Recommendation System](https://www.kaggle.com/code/alihassanml/fashion-recomended-system)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alihassanml/Fashion-Product-Recomended-System.git
   cd Fashion-Product-Recomended-System
   ```

2. **Install required dependencies:**
   Make sure you have Python 3.8 or above installed. Install the dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Required Files:**
   - Place `styles.csv` in the root directory.
   - Ensure `feature_data.pkl` and `file_names.pkl` are available in the root directory.
   - Place all images in the `./images/` folder.

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Upload an Image:**
   - Click on "Choose an image..." to upload an image of a fashion product.
   - The app will display similar products with relevant metadata.

## Project Structure

```plaintext
Fashion-Product-Recomended-System/
│
├── app.py                     # Main Streamlit app script
├── styles.csv                 # Metadata CSV file for fashion products
├── feature_data.pkl           # Pickle file containing extracted image features
├── file_names.pkl             # Pickle file containing image filenames
├── images/                    # Directory containing all fashion product images
├── requirements.txt           # List of required dependencies
└── README.md                  # Project documentation
```

## Dependencies

- Python 3.8+
- Streamlit
- TensorFlow
- Pandas
- Numpy
- Scikit-learn
- Matplotlib

You can install the dependencies using the command:
```bash
pip install -r requirements.txt
```

## How It Works

1. **Feature Extraction**: Uses a pre-trained ResNet50 model to extract features from the uploaded image.
2. **Similarity Search**: The app finds the closest matches using the Nearest Neighbors algorithm.
3. **Display Results**: Similar images are displayed along with their metadata like gender, article type, and season.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Ali Hassan**  
GitHub: [alihassanml](https://github.com/alihassanml)  
Project Link: [Fashion Product Recommendation System](https://github.com/alihassanml/Fashion-Product-Recomended-System)
