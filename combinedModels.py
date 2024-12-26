#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

print("✓ Librairies importées avec succès")

# Configuration de base
EMBEDDING_DIM = 20
MLP_LAYERS = [50, 20]
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

print("✓ Configuration de base initialisée")


# In[3]:


class MetaNCF:
    def __init__(self, criterion_models):
        print("\nInitialisation du meta-modèle")
        self.criterion_models = criterion_models
        input_size = len(criterion_models)
        print(f"Nombre de modèles à combiner: {input_size}")
        
        self.meta_model = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.meta_model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

        # Nouveaux attributs pour le suivi des pertes
        self.batch_losses = []
        self.epoch_losses = []
        
    def get_criterion_predictions(self, user_ids, item_ids):
        predictions = []
        for idx, model in enumerate(self.criterion_models):
            pred = model.predict(user_ids, item_ids)
            predictions.append(pred.unsqueeze(1))
        return torch.cat(predictions, dim=1)
    
    def train(self, train_loader, num_epochs=NUM_EPOCHS):
        print("\nDébut de l'entraînement du meta-modèle")
        # Réinitialiser les listes de pertes
        self.batch_losses = []
        self.epoch_losses = []
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
                self.optimizer.zero_grad()
                criterion_preds = self.get_criterion_predictions(user_ids, item_ids)
                final_predictions = self.meta_model(criterion_preds)
                loss = self.criterion(final_predictions.squeeze(), ratings)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Meta Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            self.epoch_losses.append(avg_loss)
            print(f"Meta Epoch {epoch+1}/{num_epochs} terminée - Loss moyenne: {avg_loss:.4f}")
    
    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            criterion_preds = self.get_criterion_predictions(user_ids, item_ids)
            return self.meta_model(criterion_preds).squeeze()

    def plot_training_loss(self):
        """
        Visualise les pertes d'entraînement du meta-modèle.
        """
        plt.figure(figsize=(12, 5))
        
        # Graphique des pertes moyennes par époque
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_losses)
        plt.title('Perte Moyenne par Époque')
        plt.xlabel('Époque')
        plt.ylabel('Perte Moyenne')
        
        plt.tight_layout()
        plt.show()

print("✓ Classe MetaNCF définie avec visualisation des pertes")


# In[4]:


class SingleCriterionNCF:
    def __init__(self, num_users, num_items, criterion_name):
        self.model = NCF(num_users, num_items)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE,weight_decay=1e-5)
        self.criterion_name = criterion_name
        self.best_loss = float('inf')
        self.patience = 3
        self.counter = 0

        # Nouveaux attributs pour le suivi des pertes
        self.train_batch_losses = []
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        
        print(f"\nModèle initialisé pour le critère: {criterion_name}")


    def train(self, train_loader, val_loader=None, num_epochs=NUM_EPOCHS):
        # Réinitialiser les listes de pertes
        self.train_batch_losses = []
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        for epoch in range(num_epochs):
            # Train phase
            self.model.train()
            total_loss = 0
            num_batches = 0
        
            for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
                self.optimizer.zero_grad()
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
            avg_loss = total_loss / num_batches
            self.train_epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs} terminée - Loss moyenne: {avg_loss:.4f}")
        
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = self._validate(val_loader)
                self.val_epoch_losses.append(val_loss)
            
                # Early stopping
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.counter = 0
                    # Sauvegarder le meilleur modèle
                    torch.save(self.model.state_dict(), f'{self.criterion_name}_best_model.pkl')
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        print(f"Early stopping pour {self.criterion_name}")
                        break
    
    def _validate(self, val_loader):
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in val_loader:
                predictions = self.model(user_ids, item_ids)
                val_loss = self.criterion(predictions, ratings)
                total_val_loss += val_loss.item()
                num_val_batches += 1
        
        return total_val_loss / num_val_batches


    def predict(self, user_ids, item_ids):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(user_ids, item_ids)
        return predictions

    def plot_training_loss(self):
        """
        Visualise les pertes d'entraînement du modèle individuel.
        """
        plt.figure(figsize=(15, 5))
        
        
        # Graphique des pertes moyennes d'entraînement par époque
        plt.subplot(1, 3, 2)
        plt.plot(self.train_epoch_losses)
        plt.title(f'Perte Moyenne d\'Entraînement\n{self.criterion_name}')
        plt.xlabel('Époque')
        plt.ylabel('Perte Moyenne')
        
        # Graphique des pertes de validation par époque
        if self.val_epoch_losses:
            plt.subplot(1, 3, 3)
            plt.plot(self.val_epoch_losses)
            plt.title(f'Perte de Validation\n{self.criterion_name}')
            plt.xlabel('Époque')
            plt.ylabel('Perte de Validation')
        
        plt.tight_layout()
        plt.show()

print("✓ Classe SingleCriterionNCF définie avec visualisation des pertes")

# print("✓ Classe SingleCriterionNCF définie")


# In[5]:


# Chargement des données
print("Chargement des données...")

df = pd.read_csv('C:/Users/HP/Desktop/IAII_M2/SystèmeDeRecommendation/projet/RSData/maData/reviews.csv')
print(f"✓ Données chargées: {len(df)} lignes")

# Encodage des utilisateurs et items
print("\nEncodage des IDs...")
le_user = LabelEncoder()
le_item = LabelEncoder()

df['user_encoded'] = le_user.fit_transform(df['author_id'])
df['item_encoded'] = le_item.fit_transform(df['offering_id'])

print(f"Nombre d'utilisateurs uniques: {len(le_user.classes_)}")
print(f"Nombre d'items uniques: {len(le_item.classes_)}")

# Définition des critères
criteria = [ 'overall_rating','service_rating', 'cleanliness_rating', 'value_rating', 
            'location_rating', 'sleep_quality_rating', 'rooms_rating']

scaler = MinMaxScaler()
for criterion in criteria:
    df[criterion] = scaler.fit_transform(df[[criterion]])
    

print(f"\nCritères à analyser: {criteria}")


# In[6]:


class RatingDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=EMBEDDING_DIM, layers=MLP_LAYERS):
        super(NCF, self).__init__()
        
        print(f"Initialisation du modèle NCF avec:")
        print(f"- Nombre d'utilisateurs: {num_users}")
        print(f"- Nombre d'items: {num_items}")
        print(f"- Dimension d'embedding: {embedding_dim}")
        print(f"- Couches MLP: {layers}")
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_dim, layer_size))
            self.dropout_layers.append(nn.Dropout(0.5))
            input_dim = layer_size
            
        self.output_layer = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        
        for layer, dropout_layer in zip(self.fc_layers, self.dropout_layers):
            vector = nn.functional.relu(layer(vector))
            vector = dropout_layer(vector)
            
        output = self.sigmoid(self.output_layer(vector))
        return output.squeeze()

print("✓ Classes RatingDataset et NCF définies")


# In[11]:


class RecommendationSystem:
    def __init__(self, meta_model, le_user, le_item, criterion_models):
        self.meta_model = meta_model
        self.le_user = le_user
        self.le_item = le_item
        self.criterion_models = criterion_models
    
    def get_top_recommendations(self, user_id, n_recommendations=5, return_scores=False):
        """
        Générer des recommandations pour un utilisateur donné, avec scores optionnels.
        """
        encoded_user_id = self.le_user.transform([user_id])[0]
        all_items = np.arange(len(self.le_item.classes_))
        
        user_tensor = torch.tensor([encoded_user_id] * len(all_items), dtype=torch.long)
        items_tensor = torch.tensor(all_items, dtype=torch.long)
        
        with torch.no_grad():
            predictions = self.meta_model.predict(user_tensor, items_tensor).numpy()
        
        top_indices = predictions.argsort()[-n_recommendations:][::-1]
        recommended_items = self.le_item.inverse_transform(top_indices)
        scores = predictions[top_indices]
        
        if return_scores:
            return list(zip(recommended_items, scores))
        return list(recommended_items)
    
    def save_model(self, path='meta_meta_ncf_recommendation_model.pkl'):
        """
        Sauvegarder le modèle meta NCF
        """
        state = {
            'meta_model_state': self.meta_model.meta_model.state_dict(),
            'le_user': self.le_user,
            'le_item': self.le_item,
            'criterion_models_states': [model.model.state_dict() for model in self.criterion_models]
        }
        torch.save(state, path)
        print(f"Modèle sauvegardé avec succès à {path}")
    
    @classmethod
    def load_model(cls, path, meta_model_class, criterion_models_class):
        """
        Charger un modèle meta NCF sauvegardé
        
        Args:
            path (str): Chemin du modèle sauvegardé
            meta_model_class (class): Classe du meta-modèle (MetaNCF)
            criterion_models_class (class): Classe des modèles individuels (SingleCriterionNCF)
        """
        state = torch.load(path, weights_only=False)
        

        # Restaurer les modèles individuels
        criterion_models = []
        for idx in range(len(criteria)):  # Utilisez la liste 'criteria' définie précédemment
            model = criterion_models_class(len(le_user.classes_), 
                                           len(le_item.classes_), 
                                           criteria[idx])
            model.model.load_state_dict(state['criterion_models_states'][idx])
            criterion_models.append(model)
        
        # Restaurer le meta-modèle
        meta_model = meta_model_class(criterion_models)
        meta_model.meta_model.load_state_dict(state['meta_model_state'])
        
        # Créer l'instance du système de recommandation
        recommendation_system = cls(meta_model, state['le_user'], state['le_item'], criterion_models)
        
        print(f"Modèle chargé avec succès depuis {path}")
        return recommendation_system

# Chargement et exploitation du modèle sauvegardé
print("\nChargement du modèle sauvegardé pour des recommandations avec scores :")
loaded_recommendation_system = RecommendationSystem.load_model(
    'C:/Users/HP/Desktop/IAII_M2/SystèmeDeRecommendation/projet/combinedModel/meta_meta_ncf_recommendation_model.pkl', 
    MetaNCF,
    SingleCriterionNCF
)

# Exemple d'utilisation avec scores
example_user_id = df['author_id'].iloc[2]  # Premier utilisateur comme exemple
recommendations_with_scores = loaded_recommendation_system.get_top_recommendations(
    example_user_id, 
    n_recommendations=5, 
    return_scores=True
)

print(f"\nTop recommandations pour l'utilisateur {example_user_id} avec scores:")
for item, score in recommendations_with_scores:
    print(f"Item: {item}, Score: {score:.4f}")


# In[8]:


def create_id_mapping(model_data, offerings_df, reviews_df):
    # Créer un mapping des IDs internes vers les IDs réels
    raw_to_inner = {}
    inner_to_raw = {}
    
    for criterion in model_data['criteria']:
        trainset = model_data['trainsets'][criterion]
        for raw_id in reviews_df['offering_id'].unique():
            try:
                inner_id = trainset.to_inner_iid(raw_id)
                raw_to_inner[raw_id] = inner_id
                inner_to_raw[inner_id] = raw_id
            except:
                continue
    
    return raw_to_inner, inner_to_raw

def get_recommendations_with_mapping(user_id, model_data, offerings_df, reviews_df, n_recommendations=5):
    _, inner_to_raw = create_id_mapping(model_data, offerings_df, reviews_df)
    
    recommendations = []
    for inner_id in range(len(inner_to_raw)):
        if inner_id not in inner_to_raw:
            continue
            
        real_id = inner_to_raw[inner_id]
        if real_id not in set(offerings_df['id']):
            continue
            
        scores = {}
        weighted_score = 0
        
        for criterion in model_data['criteria']:
            try:
                prediction = model_data['models'][criterion].predict(user_id, inner_id)
                scores[criterion] = prediction.est
                weighted_score += prediction.est * model_data['weights'][criterion]
            except:
                continue
        
        if weighted_score > 0:
            hotel_info = offerings_df[offerings_df['id'] == real_id].iloc[0].to_dict()
            hotel_reviews = reviews_df[reviews_df['offering_id'] == real_id]
            
            avg_ratings = {
                criterion: hotel_reviews[criterion].mean() 
                for criterion in model_data['criteria'] 
                if not hotel_reviews[criterion].isna().all()
            }
            
            recommendations.append({
                'hotel_id': real_id,
                'weighted_score': weighted_score,
                'predicted_scores': scores,
                'avg_actual_ratings': avg_ratings,
                'hotel_info': hotel_info
            })
    
    recommendations.sort(key=lambda x: x['weighted_score'], reverse=True)
    return recommendations[:n_recommendations]


# In[14]:


import cloudpickle
import pickle
import dill
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Configuration de base (identique à votre configuration)
EMBEDDING_DIM = 20
MLP_LAYERS = [50, 20]
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=EMBEDDING_DIM, layers=MLP_LAYERS):
        super(NCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_dim, layer_size))
            self.dropout_layers.append(nn.Dropout(0.5))
            input_dim = layer_size
            
        self.output_layer = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        
        for layer, dropout_layer in zip(self.fc_layers, self.dropout_layers):
            vector = nn.functional.relu(layer(vector))
            vector = dropout_layer(vector)
            
        output = self.sigmoid(self.output_layer(vector))
        return output.squeeze()

class HybridHotelRecommender:
    def __init__(self, models_path: Dict[str, str], weights: Dict[str, float] = None):
        self.models_path = models_path
        self.weights = weights or {
            'ncf': 0.44,
            'cf': 0.21,
            'cbf': 0.21,
            'sentiment': 0.14
        }
        
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        print("Loading models...")
        self.load_models()
        
    def save_model(self, filepath: str):
        """
        Sauvegarde le modèle hybride entier dans un fichier.
        """
        # Sauvegarde des modèles spécifiques non sérialisables
        torch.save(self.meta_model.meta_model.state_dict(), filepath + '_meta_model.pkl')
        for idx, model in enumerate(self.criterion_models):
            torch.save(model.model.state_dict(), filepath + f'_criterion_{idx}.pkl')
        
        # Sauvegarde des données sérialisables
        with open(filepath + '_main.pkl', 'wb') as f:
            dill.dump({
                'models_path': self.models_path,
                'weights': self.weights,
                'sentiment_data': self.sentiment_data,
                'cf_model': self.cf_model,
                'cbf_model': self.cbf_model
            }, f)
        print(f"Hybrid model saved to {filepath}")
        
    @staticmethod
    def load_model(filepath: str) -> "HybridHotelRecommender":
        """
        Charge le modèle hybride depuis un fichier.
        """
        # Charger les données sérialisables
        with open(filepath + '_main.pkl', 'rb') as f:
            data = dill.load(f)
        
        recommender = HybridHotelRecommender(data['models_path'], data['weights'])
        recommender.sentiment_data = data['sentiment_data']
        recommender.cf_model = data['cf_model']
        recommender.cbf_model = data['cbf_model']
        
        # Recharger les modèles spécifiques non sérialisables
        recommender.meta_model.meta_model.load_state_dict(torch.load(filepath + '_meta_model.pkl'))
        for idx, model in enumerate(recommender.criterion_models):
            model.model.load_state_dict(torch.load(filepath + f'_criterion_{idx}.pkl'))
        
        print(f"Hybrid model loaded from {filepath}")
        return recommender

    def load_models(self):
        """
        Charge tous les modèles avec leurs méthodes de chargement spécifiques
        """
        # Charger le modèle NCF
        print("Loading NCF model...")
        state = torch.load(self.models_path['ncf'], weights_only=False)
        self.le_user = state['le_user']
        self.le_item = state['le_item']
        
        # Initialiser les modèles de critère
        self.criterion_models = []
        for idx in range(len(criteria)):
            model = SingleCriterionNCF(
                len(self.le_user.classes_),
                len(self.le_item.classes_),
                criteria[idx]
            )
            model.model.load_state_dict(state['criterion_models_states'][idx])
            self.criterion_models.append(model)
        
        # Initialiser et charger le meta-modèle
        self.meta_model = MetaNCF(self.criterion_models)
        self.meta_model.meta_model.load_state_dict(state['meta_model_state'])
        self.ncf_recommender = RecommendationSystem(
            self.meta_model,
            self.le_user,
            self.le_item,
            self.criterion_models
        )
        
        # Charger le modèle CF
        print("Loading CF model...")
        with open(self.models_path['cf'], 'rb') as f:
            self.cf_model = pickle.load(f)
            
        # Charger le modèle CBF
        print("Loading CBF model...")
        with open(self.models_path['cbf'], 'rb') as f:
            self.cbf_model = cloudpickle.load(f)
            
        # Charger le modèle de sentiment
        print("Loading Sentiment model...")
        with open(self.models_path['sentiment'], 'rb') as f:
            model_data = dill.load(f)
            self.sentiment_model = model_data['recommender']
            self.sentiment_data = {
                'df_sampled': model_data['df_sampled'],
                'offerings_df': model_data['offerings_df'],
                'ratings_similarity': model_data['ratings_similarity'],
                'reviews_similarity': model_data['reviews_similarity'],
                'sentiment_similarity': model_data['sentiment_similarity'],
                'original_df': model_data['original_df']
            }

    def get_ncf_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Obtient les recommandations du modèle NCF
        """
        return self.ncf_recommender.get_top_recommendations(
            user_id,
            n_recommendations=n_recommendations,
            return_scores=True
        )

    def get_cf_recommendations(self, user_id: str, user_data: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Obtient les recommandations du modèle CF
        """
        recommendations = get_recommendations_with_mapping(
            user_id,
            self.cf_model,
            self.offerings_df,
            user_data
        )
        return [(str(rec['hotel_id']), rec['weighted_score']) for rec in recommendations]

    def get_cbf_recommendations(self, user_id: str, user_data: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Obtient les recommandations du modèle CBF
        """
        recommendations = self.cbf_model.recommend_for_user(user_id, user_data)
        return [(str(row['hotel_id']), row['similarity_score']) 
                for _, row in recommendations.iterrows()]

    def get_sentiment_recommendations(self, user_id: str, user_data: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Obtient les recommandations du modèle de sentiment
        """
        # Recherche du nom d'utilisateur correspondant à l'user_id dans user_data
        user_data_row = user_data[user_data['author_id'] == user_id]
        
        if user_data_row.empty:
            raise ValueError(f"User ID {user_id} not found in the user data.")
        print(f"Username found: {user_data_row.iloc[0]['author_username']}")
        
        # Passer le nom d'utilisateur à la méthode de recommandation
        try:
            recommendations = self.sentiment_model.recommend_for_user(
                self.sentiment_data['df_sampled'],
                self.sentiment_data['offerings_df'],
                self.sentiment_data['ratings_similarity'],
                self.sentiment_data['reviews_similarity'],
                self.sentiment_data['sentiment_similarity'],
                user_data_row.iloc[0]['author_username'],  
                self.sentiment_data['original_df'],
            )
    
            print(f"Recommendations: {recommendations}")
            return [(str(row.name), row['similarity_score']) 
                    for _, row in recommendations.iterrows()]
        
        except Exception as e:
            print(f"Error during sentiment recommendations: {str(e)}")
            raise



    def normalize_scores(self, recommendations: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Normalise les scores entre 0 et 1
        """
        if not recommendations:
            return []
        
        scores = [score for _, score in recommendations]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [(item, 1.0) for item, _ in recommendations]
        
        return [(item, (score - min_score) / (max_score - min_score)) 
                for item, score in recommendations]

    def get_hybrid_recommendations(self, 
                                 user_id: str, 
                                 user_data: pd.DataFrame,
                                 offerings_df: pd.DataFrame,
                                 n_recommendations: int = 5) -> pd.DataFrame:
        """
        Obtient les recommandations hybrides combinées
        """
        self.offerings_df = offerings_df
        
        # Obtenir toutes les recommandations
        ncf_recs = self.normalize_scores(self.get_ncf_recommendations(user_id))
        cf_recs = self.normalize_scores(self.get_cf_recommendations(user_id, user_data))
        cbf_recs = self.normalize_scores(self.get_cbf_recommendations(user_id, user_data))
        sentiment_recs = self.normalize_scores(self.get_sentiment_recommendations(user_id, user_data))
        
        # Combiner les scores
        combined_scores = {}
        
        for model_recs, weight_key in [
            (ncf_recs, 'ncf'),
            (cf_recs, 'cf'),
            (cbf_recs, 'cbf'),
            (sentiment_recs, 'sentiment')
        ]:
            for item, score in model_recs:
                if item not in combined_scores:
                    combined_scores[item] = 0
                combined_scores[item] += score * self.weights[weight_key]
        
        # Obtenir les top N recommandations
        top_items = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        # Créer le DataFrame final
        recommendations_df = pd.DataFrame(
            top_items,
            columns=['hotel_id', 'hybrid_score']
        )
        recommendations_df = recommendations_df.merge(
            offerings_df[['id', 'name', 'hotel_class', 'locality']],
            left_on='hotel_id',
            right_on='id',
            how='left'
        )
        
        return recommendations_df[['hotel_id', 'name', 'hotel_class', 'locality', 'hybrid_score']]

# Fonction d'exemple d'utilisation
def main():
    # Chemins des modèles
    models_path = {
        'ncf': 'C:/Users/HP/Desktop/IAII_M2/SystèmeDeRecommendation/projet/combinedModel/meta_meta_ncf_recommendation_model.pkl',
        'cf': 'C:/Users/HP/Desktop/IAII_M2/SystèmeDeRecommendation/projet/combinedModel/svd_recommendation_model.pkl',
        'cbf': 'C:/Users/HP/Desktop/IAII_M2/SystèmeDeRecommendation/projet/combinedModel/hotel_recommender1.pkl',
        'sentiment': 'C:/Users/HP/Desktop/IAII_M2/SystèmeDeRecommendation/projet/combinedModel/sa3_model.pkl'
    }
    
    # Charger les données nécessaires
    # offerings_df = pd.read_csv('/kaggle/input/tripad2/offering.csv')
    
    offerings_df = pd.read_csv('C:/Users/HP/Desktop/IAII_M2/SystèmeDeRecommendation/projet/RSData/maData/offering.csv')
    user_data = pd.read_csv('C:/Users/HP/Desktop/IAII_M2/SystèmeDeRecommendation/projet/RSData/maData/reviews.csv')
    
    # Créer l'instance du recommandeur hybride
    hybrid_recommender = HybridHotelRecommender(models_path)
    
    # Exemple d'utilisation
    user_id = "FB1032DECE1162CB3556D05F278AAFFD"
    recommendations = hybrid_recommender.get_hybrid_recommendations(
        user_id,
        user_data,
        offerings_df
    )
    
    print("\nRecommandations hybrides:")
    print(recommendations)
    
    # Sauvegarder le modèle
    hybrid_recommender.save_model("hybrid_recommender_model.pkl")
    loaded_recommender = HybridHotelRecommender.load_model("hybrid_recommender_model.pkl")
    print("Model successfully saved and loaded!")
    
    # Obtenir des recommandations hybrides
    recommendations = loaded_recommender.get_hybrid_recommendations(
        user_id=user_id,
        user_data=user_data,
        offerings_df=offerings_df,
        n_recommendations=10
    )

    # Afficher les recommandations
    print("Recommandations apres le sauvgardage:")
    print(recommendations)


if __name__ == "__main__":
    main()

