from flask import Flask,jsonify,request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId


app =  Flask(__name__)
CORS(app)


client = MongoClient('mongodb+srv://shiv_test:test@cluster0.xzzdmgf.mongodb.net/test')
db = client["craft"]
collect = db["products"]

data = []
for doc in collect.find():
     data.append({
            'id': str(doc['_id']),
            'name': doc['name'],
            'price': doc['price'],
            'category': doc['category'],
            'image_link': doc['image_link']
     })


df = pd.DataFrame(data)
X1 = df["name"].fillna(" ")
# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X1)
# Compute cosine similarity between products
cosine_similarities = cosine_similarity(X)

@app.route("/recommand", methods = ['POST'] )
def recommand():
     recProducts = []
     data = request.json
     name = data['name']
     print(name)

     #find name in dataframe and get its index number
     product_index =  df.index[df['name'] == name].tolist()[0]
     # Get top 5 most similar products
     similar_products_indices = cosine_similarities[product_index].argsort()[:-6:-1][0:]
     similar_products = df.iloc[similar_products_indices]
     # make a list and paste all the products in it
     for i, product in similar_products.iterrows():
        recProducts.append({
            'id': product['id'],
            'name': product['name'],
            'price': product['price'],
            'category': product['category'],
            'image_link': product['image_link']
         })
     # return the list
     return jsonify(recProducts)

if __name__ == "__main__":
        app.run()