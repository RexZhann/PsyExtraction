from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
from tqdm import tqdm
from dashscope import Generation
from.embedding import generate_embeddings


def plot_heatmap(related_phrases_embeddings, related_phrases, keyword):
    similarity_matrix = cosine_similarity(related_phrases_embeddings)

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=related_phrases, yticklabels=related_phrases, cmap='coolwarm')
    plt.title(f'Related Phrases Similarity Heatmap for word {keyword}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

    return similarity_matrix


def plot_H_cluster(similarity_matrix, keyword):
    # 层次聚类
    Z = linkage(similarity_matrix, 'ward')

    # 绘制层次聚类树状图
    plt.figure(figsize=(10, 8))
    dendrogram(Z)
    plt.title(f'Hierarchical Clustering Dendrogram for {keyword}')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()




def get_rep_phrases_target(similarity_matrix, related_phrases, related_phrases_embeddings, target_emb, k=6):
    rep_phrases = []
    rep_phrase_emb = []
    # 使用相似度矩阵进行层次聚类
    Z = linkage(1 - similarity_matrix, 'ward')  # 使用1减去相似度矩阵作为距离矩阵
    # 将短语分为k类
    clusters = fcluster(Z, k, criterion='maxclust')
    # 计算target_emb与所有短语嵌入的相似度
    target_similarity = np.dot(related_phrases_embeddings, target_emb)
    
    for cluster_id in range(1, k + 1):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        
        # 如果当前聚类中没有元素，则跳过
        if not cluster_indices:
            continue
        
        # 计算target_emb与当前聚类中所有短语的相似度
        cluster_similarities = target_similarity[cluster_indices]
        
        # 找出与target_emb最相似的短语索引
        max_similarity_index = np.argmax(cluster_similarities)
        rep_index = cluster_indices[max_similarity_index]
        
        # 添加代表性短语和其嵌入向量
        rep_phrases.append(related_phrases[rep_index])
        rep_phrase_emb.append(related_phrases_embeddings[rep_index])
        
    return rep_phrases, rep_phrase_emb


def get_rep_phrases_average(similarity_matrix, clusters, related_phrases, related_phrases_embeddings, k=6):
    # 使用相似度矩阵进行层次聚类
    Z = linkage(1 - similarity_matrix, 'ward')  # 使用1减去相似度矩阵作为距离矩阵

    clusters = fcluster(Z, k, criterion='maxclust')
    rep_phrases = []
    rep_phrase_emb = []

    for cluster_id in range(1, k + 1):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        cluster_similarity = similarity_matrix[cluster_indices][:, cluster_indices]
        centroid_index = np.argmax(np.sum(cluster_similarity, axis=0))
        rep_index = cluster_indices[centroid_index]
        rep_phrases.append(related_phrases[rep_index])
        rep_phrase_emb.append(related_phrases_embeddings[rep_index])

    return rep_phrases, rep_phrase_emb


def average_dot_product(vectors):
    n = len(vectors)
    dot_products = []

    for i in range(n):
        for j in range(i + 1, n):
            dot_product = np.dot(vectors[i], vectors[j])
            dot_products.append(dot_product)
    
    average_dot = np.mean(dot_products)
    return average_dot


def average_cos_sim(vectors):
    return np.mean(cosine_similarity(vectors))


def retriever_batch(articles, final_phrases, batchsize=6, modl='qwen-max'):
    def batch_generator(articles, batchsize):
        for i in range(0, len(articles), batchsize):
            yield articles[i:i + batchsize]
    
    retrieval_res = []
    
    with tqdm(total=len(articles) // batchsize + (len(articles) % batchsize > 0)) as pbar:
        for idx, batch in enumerate(batch_generator(articles, batchsize)):
            
            debug_info = f"Processing batch {idx + 1}/{len(articles) // batchsize + (len(articles) % batchsize > 0)} with {len(batch)} phrases"
            
            print(f"Debug: {debug_info}")  
            
            prompt = f'''complete the #OBJECTIVE# based on the #CONTEXT#, and generate the output based on #STYLE# and #RESPONSE#
            # CONTEXT #
            You are an expert at retrieving a phrase from a text database given a list of the characteristics of that phrase.
            # OBJECTIVE #
            Pick exactly one phrase or word from the text database that best fits the list of characteristics.

            list of characteristics: {final_phrases}

            text database: {batch}

            # STYLE #
            output only the result of phrase retrieval. 
            The result could be a single word or a short phrase, but never a sentence. 
            The result must be from the text database.
            Set the temperature parameter to 0 to ensure a precise output
            # RESPONSE #
            do not output anything other than the result of retrieval
            '''
            
            rsp = Generation.call(model=modl, prompt=prompt)
            retrieval_res.append(rsp.output.text.strip())

            pbar.update(len(batch))
     
    return retrieval_res


def eval_retrieval_llm(retrieval_res, keyword, collection):
    ave_emb_1 = np.array([generate_embeddings(phrase) for phrase in retrieval_res])
    retrieval_1 = [doc.fields['title'] for doc in collection.query(np.mean(ave_emb_1, axis=0), output_fields=['title'],topk=1).output]
    cos_score = np.dot(np.mean(ave_emb_1, axis=0), generate_embeddings(keyword)) / (np.linalg.norm(np.mean(ave_emb_1, axis=0)) * np.linalg.norm(generate_embeddings(keyword)))

    return retrieval_1, cos_score

def eval_retrieval_que(final_emb, keyword, collection):
    retrieval_2 = [doc.fields['title'] for doc in collection.query(np.mean(np.array(final_emb), axis=0), output_fields=['title'],topk=1).output]
    cos_score = np.dot(np.mean(final_emb, axis=0), generate_embeddings(keyword)) / (np.linalg.norm(np.mean(final_emb, axis=0)) * np.linalg.norm(generate_embeddings(keyword)))

    return retrieval_2, cos_score