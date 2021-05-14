from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "Official ICC Cricket website - live matches, scores, news, highlights, commentary, rankings",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "Live Cricket Score, Ball by Ball Commentary, Scorecard Updates, Match Facts & related News"
]
model_name='bert-base-nli-mean-tokens'

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embeddings = model.encode(sentences)
sentence_vect=model.encode(sentences)
#print(sentence_vect)
print(sentence_embeddings.shape)

print(cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
))