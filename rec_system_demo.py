import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#only use movies with a rating > 5.0
data = fetch_movielens(min_rating = 5.0)

#model creation
model = LightFM(loss = 'warp')

#model training
model.fit(data['train'], epochs = 30, num_threads=2)

def sample_recommendation(model, data, user_ids):

	#get number of users and movies
	n_users, n_items = data['train'].shape

	#make new recommendations for given user
	for user_id in user_ids:
		#gather movies already known be liked by the user
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#movie predictions that the user might like
		scores = model.predict(user_id, np.arange(n_items))

		#organize results from best to worst
		top_items = data['item_labels'][np.argsort(-scores)]

		#print results
		print("User %s" % user_id)
		print("Known positives:")

		for x in known_positives[:3]:
			print("%s" % x)
		print("Recommended:")

		#Print top 3 suggestions
		for x in top_items[:3]:
			print("%s" % x)

#Enter 3 users to get recommendations
sample_recommendation(model, data, [190, 5, 220])
