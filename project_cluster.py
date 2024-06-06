import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def find_optimal_clusters(data, max_k):
    iters=range(2, max_k, 1)
    sse=[]
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=42).fit(data).inertia_)
        print('Fit {} clusters'.format(k))

    f, ax = plt.subplots(1,1)
    ax.plot(iters, sse, marker= 'o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    df_terms = pd.DataFrame()
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
        df_terms[i] = [labels[t] for t in np.argsort(r)[-n_terms:]]
    save_file = 'kmeans_' + str(len(df)) + '_topics_keywords.csv'
    df_terms.to_csv(save_file, index=False)


file = 'pay_day_2.csv'
df = pd.read_csv(file)
tfidf = TfidfVectorizer()
df = df[df['review_text'].notna()]
text = tfidf.fit_transform(df['review_text'])

# find_optimal_clusters(text, 20)

clusters = KMeans(n_clusters=8, random_state=42).fit_predict(text)
get_top_keywords(text, clusters, tfidf.get_feature_names_out(), 20)
df['kmeans_cluster'] = clusters
save_file = file.split('.')[0] + '_kmeans_cluster.csv'
df.to_csv(save_file, index=False)