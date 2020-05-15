import tensorflow as tf
import pandas as pd

from rbm_model import RBM

# Preprocessing
ratings_df = pd.read_csv('./data/ratings.dat', sep='::', header=None, engine='python')

ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

user_rating_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating')
norm_user_rating_df = user_rating_df.fillna(0)/5.0

train_x, test_x = tf.split(norm_user_rating_df.to_numpy().astype('float32'), 2)

train_dataset = tf.data.Dataset.from_tensor_slices(train_x)
train_dataset = train_dataset.shuffle(buffer_size=3020).batch(20)

test_dataset = tf.data.Dataset.from_tensor_slices(test_x)
test_dataset = test_dataset.shuffle(buffer_size=3020).batch(20)

model = RBM(len(user_rating_df.columns), 128)

epochs = 12
alpha = 0.01

def run(dataset, train=True):
    for step, batch in enumerate(dataset):
        loss = model.train(batch, alpha) if train else model.evaluate(batch)
        
        if step % 10 == 0:
            print('step %s: loss = %s' % (step, loss))

for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))
    run(train_dataset)

# Evaluation
run(test_dataset, train=False)

# Test the model with a mock case
mock_user_id = 175

inputUser = tf.reshape(train_x[mock_user_id - 1], (1, -1))
recommended = model.predict(inputUser)

movies_df = pd.read_csv('./data/movies.dat', sep='::', header=None, engine='python')
movies_df.columns = ['MovieID', 'Title', 'Genres']

# List of movies recommended to user 175
scored_movies_df_mock = movies_df[movies_df['MovieID'].isin(user_rating_df.columns)]
scored_movies_df_mock = scored_movies_df_mock.assign(RecommendationScore = recommended[0])

print(scored_movies_df_mock.sort_values(["RecommendationScore"], ascending=False).head())
