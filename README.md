# Hybrid Hotel Recommendation System

## Overview
This project implements a hybrid recommendation system for hotels by combining multiple recommendation techniques:

1. **Neural Collaborative Filtering (NCF):** Predicts user preferences based on historical interactions.
2. **Collaborative Filtering (CF):** Utilizes similarities between users and items to generate recommendations.
3. **Content-Based Filtering (CBF):** Recommends hotels based on item features (e.g., class, locality).
4. **Sentiment Analysis:** Enhances recommendations by incorporating user reviews and sentiments.

The hybrid system combines these approaches to provide more accurate and personalized recommendations.

---

## Files
- **`hybrid_recommender.py`**: Contains the implementation of the recommendation system.
- **Model Files**: Serialized models used by the system (e.g., `ncf_model.pkl`, `cf_model.pkl`, etc.). Ensure paths to these files are correctly configured.

---

## Dependencies
The following Python libraries are required:

```bash
pip install pandas numpy torch scikit-learn
```

---

## Features

### Neural Collaborative Filtering (NCF)
- Uses a pre-trained neural network model to predict user-item interactions.
- Model is loaded from `ncf_model.pkl`.

### Collaborative Filtering (CF)
- Employs matrix factorization techniques to find similar users/items.
- Model is loaded from `cf_model.pkl`.

### Content-Based Filtering (CBF)
- Matches users to items based on item features such as hotel class and locality.
- Model is loaded from `cbf_model.pkl`.

### Sentiment Analysis
- Analyzes user reviews to provide sentiment-based recommendations.
- Model is loaded from `sentiment_model.pkl`.

---

## Code Explanation

### Key Functions

#### `HybridHotelRecommender`
This is the main class for the recommendation system. Key methods include:

- **`load_model(self, path: str)`**: Loads a pre-trained model from the specified path.
- **`save_model(self, model, path: str)`**: Saves a model to the specified path.
- **`normalize_scores(self, scores: Dict[str, float])`**: Normalizes recommendation scores to the range [0, 1].
- **`get_hybrid_recommendations(self, user_id, user_data, offerings_df, n_recommendations=10)`**:
  Combines recommendations from NCF, CF, CBF, and Sentiment models to generate hybrid recommendations.

#### Individual Recommendation Functions
- **`get_ncf_recommendations(self, user_id, offerings_df, n_recommendations=10)`**:
  Generates recommendations using the NCF model.
- **`get_cf_recommendations(self, user_id, offerings_df, n_recommendations=10)`**:
  Generates recommendations using the CF model.
- **`get_cbf_recommendations(self, user_id, offerings_df, n_recommendations=10)`**:
  Recommends items based on content features.
- **`get_sentiment_recommendations(self, user_data, n_recommendations=10)`**:
  Uses sentiment analysis to prioritize recommendations based on user reviews.

---

## Usage

### Setting Up Models
1. Train or obtain pre-trained models for NCF, CF, CBF, and Sentiment Analysis.
2. Save the models as `.pkl` files and update the `models_path` dictionary in the `main()` function with the correct paths.

### Running the Script
Ensure your environment has all required dependencies installed and execute the following command:

```bash
python hybrid_recommender.py
```

### Example Input
The system requires the following inputs:

1. **`user_id`**: A unique identifier for the user.
2. **`user_data`**: A DataFrame containing user information (e.g., username, review data).
3. **`offerings_df`**: A DataFrame containing hotel data (e.g., id, name, hotel_class, locality).

### Output
The script outputs a list of recommended hotels for the given user, sorted by hybrid recommendation scores.

---

## Customization

### Adding Models
To add new models or techniques to the hybrid system:
1. Define a new method for generating recommendations in the `HybridHotelRecommender` class.
2. Incorporate the new method in `get_hybrid_recommendations`.

### Adjusting Weights
Modify the weighting scheme in `get_hybrid_recommendations` to prioritize certain models over others.

---

## Future Enhancements
- **Real-time Sentiment Analysis:** Integrate real-time analysis for live user feedback.
- **UI Integration:** Build a web or mobile interface for user interaction.
- **Hyperparameter Tuning:** Optimize model parameters for better accuracy.

---

## License
This project is licensed under the MIT License.

---

## Contact
For questions or suggestions, please reach out to the project maintainer.

---

