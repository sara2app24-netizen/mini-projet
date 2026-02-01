import numpy as np
from numpy.linalg import norm
# --------------------------------------------------
# Mini projet sujet 3: Modélisation d’un système de
# recommandation simple basé sur l’algèbre linéaire
# Benatmane Sara et Keidia Amina
# --------------------------------------------------

# ======================
# Classes
# ======================


class User:
    def __init__(self, user_id, name):
        self.id = user_id
        self.name = name


class Item:
    def __init__(self, item_id, title, category):
        self.id = item_id
        self.title = title
        self.category = category


class RatingMatrix:
    def __init__(self, users, items):
        self.users = users
        self.items = items
        self.matrix = np.full((len(users), len(items)), np.nan)

    def add_rating(self, user_id, item_id, value):
        self.matrix[user_id, item_id] = value

    def get_user_vector(self, user_id, category=None):
        if category:
            indices = [
                i for i, item in enumerate(self.items)
                if item.category == category
            ]
            return self.matrix[user_id, indices], indices

        return self.matrix[user_id], list(range(len(self.items)))


class RecommenderSystem:
    def __init__(self, rating_matrix):
        self.rating_matrix = rating_matrix

    # ----------------------
    # Cosine Similarity
    # ----------------------
    def cosine_similarity(self, u, v):
        mask = ~np.isnan(u) & ~np.isnan(v)
        if np.sum(mask) == 0:
            return 0.0
        u_valid = u[mask]
        v_valid = v[mask]
        return np.dot(u_valid, v_valid) / (norm(u_valid) * norm(v_valid))

    # ----------------------
    # Recommendation (Normalized + Filtering)
    # ----------------------
    def recommend_normalized(
        self,
        user_id,
        category=None,
        threshold=0.3,
        min_rating=3.0,
        exclude_items=[]
    ):

        if category is None:
            threshold = min(threshold, 0.2)
            min_rating = min(min_rating, 2.5)

        target_vector, indices = (
            self.rating_matrix.get_user_vector(user_id, category)
        )

        recommended_items = []

        for idx_pos, item_index in enumerate(indices):
            if not np.isnan(target_vector[idx_pos]):
                continue
            if item_index in exclude_items:
                continue

            weighted_sum = 0.0
            sim_sum = 0.0

            for other_id in range(len(self.rating_matrix.users)):
                if other_id == user_id:
                    continue

                other_vector, _ = (
                    self.rating_matrix.get_user_vector(other_id, category)
                )
                sim = self.cosine_similarity(target_vector, other_vector)

                if sim > threshold:
                    other_rating = other_vector[idx_pos]
                    if not np.isnan(other_rating):
                        weighted_sum += sim * other_rating
                        sim_sum += sim

            if sim_sum > 0:
                predicted_rating = weighted_sum / sim_sum

                if predicted_rating >= min_rating:
                    recommended_items.append(
                     (self.rating_matrix.items[item_index].title,
                      predicted_rating)
                                            )

        recommended_items.sort(key=lambda x: x[1], reverse=True)
        return recommended_items


# ======================
# Main
# ======================

def main():
    # users
    users = [
        User(0, "Aya"),
        User(1, "Sara"),
        User(2, "Amina"),
        User(3, "Adam"),
        User(4, "Omar")
    ]

    # items
    items = [
        Item(0, "Book A", "Book"),
        Item(1, "Book B", "Book"),
        Item(2, "Book C", "Book"),
        Item(3, "Movie X", "Movie"),
        Item(4, "Movie Y", "Movie"),
        Item(5, "Movie Z", "Movie")
    ]

    # Rating Matrix
    rm = RatingMatrix(users, items)

    # Aya's raiting
    rm.add_rating(0, 0, 5)
    rm.add_rating(0, 1, 3)
    rm.add_rating(0, 3, 4)

    # Sara's raiting
    rm.add_rating(1, 0, 4)
    rm.add_rating(1, 2, 5)
    rm.add_rating(1, 3, 5)
    rm.add_rating(1, 4, 2)

    # Amina's raiting
    rm.add_rating(2, 1, 4)
    rm.add_rating(2, 2, 4)
    rm.add_rating(2, 4, 5)
    rm.add_rating(2, 5, 3)

    # Adam's raiting
    rm.add_rating(3, 0, 2)
    rm.add_rating(3, 2, 5)
    rm.add_rating(3, 3, 5)
    rm.add_rating(3, 5, 4)

    # Omar's raiting
    rm.add_rating(4, 1, 3)
    rm.add_rating(4, 2, 2)
    rm.add_rating(4, 4, 5)
    rm.add_rating(4, 5, 5)

    recommender = RecommenderSystem(rm)

    # ======================
    # Results
    # ======================

    print("Normalized recommendations for Aya (Books):")
    for title, score in recommender.recommend_normalized(
        user_id=0,
        category="Book",
        threshold=0.3,
        min_rating=3.0
    ):
        print(f"{title} : {score:.2f}/5")

    print("\nNormalized recommendations for Aya (Movies):")
    for title, score in recommender.recommend_normalized(
        user_id=0,
        category="Movie",
        threshold=0.3,
        min_rating=3.0
    ):
        print(f"{title} : {score:.2f}/5")

    print("\nNormalized recommendations for Aya (All categories):")
    for title, score in recommender.recommend_normalized(
        user_id=0,
        category=None,
        threshold=0.3,
        min_rating=3.0
    ):
        print(f"{title} : {score:.2f}/5")


if __name__ == "__main__":
    main()
