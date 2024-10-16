# sk-None-MBnJDoVct3VyFl68qtDAT3BlbkFJuQh6D35FQEZ8wrfaJxZK

import openai
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')

# api_key = ''
client = openai.OpenAI(api_key=api_key)

KG_file_path = './dataset/knowledge graph of chronic pain_aug_f_combine_extend.xlsx'
file_path = './dataset/AI Data Set with Categories_vfL2L3_combine.csv'
embedding_save_path = './Embeddings_saved/CP_KG_embeddings'


# 定义函数去掉括号及其中的内容，并替换下划线为空格（用于文本预处理）
def preprocess_text(text):
    if pd.isna(text):
        return ''
    # 去掉括号及其中的内容
    text = re.sub(r'\(.*?\)', '', text).strip()
    text = text.replace('_', ' ')
    text = text.lower()
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    # 去除停用词
    # tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)


kg_data = pd.read_excel(KG_file_path, usecols=['subject', 'relation', 'object'])

# 创建知识图谱，保留所有节点和关系
knowledge_graph = {}
for index, row in kg_data.iterrows():
    subject = row['subject']
    relation = row['relation']
    obj = row['object']

    if subject not in knowledge_graph:
        knowledge_graph[subject] = []
    knowledge_graph[subject].append((relation, obj))

    if obj not in knowledge_graph:
        knowledge_graph[obj] = []
    knowledge_graph[obj].append((relation, subject))

# 预处理object列并过滤掉relation为is_a的行
kg_data['object_preprocessed'] = kg_data.apply(
    lambda row: preprocess_text(row['object']) if row['relation'] != 'is_a' else None,
    axis=1
)
# 仅使用预处理后的object列作为症状节点列表
symptom_nodes = kg_data['object_preprocessed'].dropna().unique().tolist()



# 获取症状节点的嵌入向量，并保存到指定路径
def get_symptom_embeddings(symptom_nodes, save_path):
    embeddings_path = os.path.join(save_path, 'KG_split_embeddings.npy')
    if os.path.exists(embeddings_path):
        print("加载已有的嵌入向量...")
        return np.load(embeddings_path)
    else:
        print("生成新的嵌入向量...")
        symptom_embeddings = []
        for symptom in tqdm(symptom_nodes):
            response = client.embeddings.create(
                input=symptom,
                model="text-embedding-3-large"
            )
            symptom_embeddings.append(response.data[0].embedding)
        np.save(embeddings_path, symptom_embeddings)

        return np.array(symptom_embeddings)


# 获取所有症状节点的嵌入向量
symptom_embeddings = get_symptom_embeddings(symptom_nodes, embedding_save_path)


def find_top_n_similar_symptoms(query, symptom_nodes, symptom_embeddings, n):
    if pd.isna(query) or not query:
        return []  # 如果查询为空或NaN，直接返回空列表
    query_preprocessed = preprocess_text(query)
    response = client.embeddings.create(
        input=query_preprocessed,
        model="text-embedding-3-large"
    )
    query_embedding = response.data[0].embedding
    if not query_embedding:
        return []  # 如果查询嵌入失败或为空，返回空列表

    if len(symptom_embeddings) > len(symptom_nodes):
        symptom_embeddings = symptom_embeddings[:len(symptom_nodes)]

    similarities = cosine_similarity([query_embedding], symptom_embeddings).flatten()

    # 获取相似度大于 0.5
    top_n_symptoms = []
    unique_symptoms = set()
    top_n_indices = similarities.argsort()[::-1]

    for i in top_n_indices:
        if similarities[i] > 0.5 and symptom_nodes[i] not in unique_symptoms:
            top_n_symptoms.append(symptom_nodes[i])
            unique_symptoms.add(symptom_nodes[i])
        if len(top_n_symptoms) == n:
            break

    return top_n_symptoms


# 计算最短路径长度（使用原始的节点名称）
def compute_shortest_path_length(node1, node2, G):
    try:
        return nx.shortest_path_length(G, source=node1, target=node2)
    except nx.NetworkXNoPath:
        return float('inf')

categories = [
    "thoracoabdominal_pain_syndromes",
    "neuropathic_pain_syndromes",
    "craniofacial_pain_syndromes",
    "cervical_spine_pain_syndromes",

    "limb_and_joint_pain_syndromes",
    "back_pain_syndromes",
    "lumbar_degenerative_and_stenosis_and_radicular_and_sciatic_syndromes",
    "generalized_pain_syndromes",

]
# 创建NetworkX图，保留完整的知识图谱结构
G = nx.Graph()
for node, edges in knowledge_graph.items():
    for relation, neighbor in edges:
        G.add_edge(node, neighbor, relation=relation)


def get_diagnoses_for_symptom(symptom):
    """
    根据症状节点，从知识图谱中获取关联的诊断节点。
    """
    diagnoses = []
    if symptom in G:
        for neighbor in G.neighbors(symptom):
            edge_data = G.get_edge_data(neighbor, symptom)
            if edge_data and 'relation' in edge_data and edge_data['relation'] != 'is_a':
                diagnoses.append(neighbor)
    return diagnoses


def find_closest_category(top_symptoms, categories,top_n):
    if isinstance(top_symptoms, pd.Series) and top_symptoms.empty:
        print("Warning: top_symptoms is empty.")
        return None
    category_votes = {category: 0 for category in categories}
    # 遍历每个症状节点
    for symptom in top_symptoms:
        top_symptoms = list(set(top_symptoms))

        # print('symptom: ',symptom)
        if symptom not in G:
            print(f"Symptom node not found in graph: {symptom}")
            continue

        diagnosis_nodes = get_diagnoses_for_symptom(symptom)
        for diagnosis in diagnosis_nodes:

            individual_diagnoses = diagnosis.split(',')

            for single_diagnosis in individual_diagnoses:
                single_diagnosis = single_diagnosis.strip().replace(' ', '_').lower()  # 去掉前后空格
                if single_diagnosis not in G:
                    print(f"Diagnosis node not found in graph: {single_diagnosis}")
                    continue

                min_distance = float('inf')
                closest_category = None

                for category in categories:
                    if category not in G:
                        print(f"Category node not found in graph: {category}")
                        continue

                    # 计算当前单独诊断到类别的最短路径距离
                    # distance = compute_shortest_path_length(single_diagnosis, category, G)
                    try:
                        distance = nx.shortest_path_length(G, source=single_diagnosis, target=category)
                    except nx.NetworkXNoPath:
                        distance = float('inf')

                    # print(f"Symptom: {symptom}, Category: {category}, Distance: {distance}")
                    if distance < min_distance:
                        min_distance = distance
                        closest_category = category

                # 单独诊断节点投票给最近的类别
                if closest_category:
                    category_votes[closest_category] += 1
    print("Category votes:", category_votes)

    sorted_categories = sorted(category_votes.items(), key=lambda x: x[1], reverse=True)
    top_n_categories = [sorted_categories[i][0] for i in range(top_n)]
    return top_n_categories


# 获取子类诊断的keyinfo值
def get_keyinfo_for_category(category, knowledge_graph):
    keyinfo_values = []
    for node, edges in knowledge_graph.items():
        if node == category:
            for relation, neighbor in edges:
                if relation == "is_a" and neighbor in knowledge_graph:
                    for rel, obj in knowledge_graph[neighbor]:
                        if rel == "has_keyinfo":
                            keyinfo_values.append(obj)
    return keyinfo_values



# 获取症状节点对应的Level 3节点（即subject）
def get_subjects_for_objects(objects, knowledge_graph):
    subjects = []
    # 处理每个object，将空格替换为下划线
    processed_objects = [obj.replace(' ', '_') for obj in objects]
    for obj in processed_objects:
        for index, row in knowledge_graph.iterrows():
            if row['object'] == obj:
                subjects.append(row['subject'])
        # print(f"Processed Object: {obj}, Subjects: {subjects}")  # 调试输出
    return subjects


# 统计症状节点连接的Level 3节点
def find_level3_for_symptoms(top_symptoms, knowledge_graph):
    level3_connections = {}
    for symptom in top_symptoms:
        # 获取与症状相关的subject节点
        subjects = get_subjects_for_objects([symptom], knowledge_graph)
        for subject in subjects:
            if subject in level3_connections:
                level3_connections[subject] += 1
            else:
                level3_connections[subject] = 1
    # print(f"Top Symptoms: {top_symptoms}, Level 3 Connections: {level3_connections}")  # 调试输出
    return level3_connections


def print_symptom_and_disease(symptom_nodes):
    """打印症状节点及其所属的疾病节点。"""
    for symptom in symptom_nodes:
        subjects = get_subjects_for_objects([symptom], kg_data)
        # print(f"Symptom: {symptom}, Disease Nodes: {subjects}")


def main_get_category_and_level3(n, participant_no,top_n):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')

    # 根据标号查找对应的行
    row = data.loc[data['Participant No.'] == str(participant_no)]
    if row.empty:
        print(f"Participant No. {participant_no} not found!")
        return None

    # 提取相关列的值
    tr = row["Level 2"].values[0]
    tr=tr.split(",")[0]
    # return [tr]

    level3real = row["Processed Diagnosis"].values[0]

    pain_location = row["Pain Presentation and Description"].values[0]
    pain_symptoms = row["Pain descriptions and assorted symptoms (self-report)"].values[0]
    pain_restriction = row["Pain restriction"].values[0]
    print(f'pain_location: {pain_location}')
    print(f'pain_symptoms: {pain_symptoms}')
    print(f'pain_restrction: {pain_restriction}')
    if pd.isna(pain_location):
        pain_location = ''
    if pd.isna(pain_symptoms):
        pain_symptoms = ''
    if pd.isna(pain_restriction):
        pain_symptoms = ''


    def process_symptom_field(field_value, symptom_nodes, symptom_embeddings, n):
        """处理症状字段并返回相应的top N相似节点列表，如果字段为空则返回空列表。"""
        if pd.isna(field_value) or field_value == '':
            return []
        return find_top_n_similar_symptoms(field_value, symptom_nodes, symptom_embeddings, n)

    top_5_location_nodes = process_symptom_field(pain_location, symptom_nodes, symptom_embeddings, n)
    top_5_symptom_nodes = process_symptom_field(pain_symptoms, symptom_nodes, symptom_embeddings, n)
    top_5_painrestriction_nodes = process_symptom_field(pain_restriction, symptom_nodes, symptom_embeddings, n)


    # 获取原始节点名称
    top_5_location_nodes_original = kg_data.loc[kg_data['object_preprocessed'].isin(top_5_location_nodes), 'object'].drop_duplicates()
    top_5_symptom_nodes_original = kg_data.loc[kg_data['object_preprocessed'].isin(top_5_symptom_nodes), 'object'].drop_duplicates()
    top_5_painrestriction_original = kg_data.loc[kg_data['object_preprocessed'].isin(top_5_painrestriction_nodes), 'object'].drop_duplicates()


    most_similar_category = find_closest_category(
        list(top_5_location_nodes_original) + list(top_5_symptom_nodes_original)+ list(top_5_painrestriction_original),
        categories,
        top_n
    )

    # print(most_similar_category)
    # print("*"*100)
    return most_similar_category


# correct_predictions = 0
# total_valid_cases = 0
# correct_level3_predictions = 0
# total_correct_category_cases = 0
#
# # 用于统计每个类中的正确和错误情况
# category_correct_count = defaultdict(int)
# category_total_count = defaultdict(int)
# category_wrong_details = defaultdict(list)
# match_n=3
# for i in tqdm(range(1,552)):
#     result = main_get_category_and_level3(match_n, i,top_n=1)
#     print(result)
#     if result:
#         data = pd.read_csv(file_path, encoding='ISO-8859-1')
#         row = data.loc[data['Participant No.'] == str(i)]
#
#         if not row.empty:
#             tr = row["Level 2"].values[0]
#             level3real = row["Processed Diagnosis"].values[0]
#
#             # 将 tr 解析为列表
#             tr_values = [item.strip() for item in tr.split(',')]
#
#             # 统计分类准确率
#             if "Unknown" not in tr_values:
#                 total_valid_cases += 1
#                 category_total_count[tr] += 1  # 统计每个类别的总数
#
#                 # 判断结果是否与真实值有任意交集
#                 if any(res in tr_values for res in result):
#                     correct_predictions += 1
#                     category_correct_count[tr] += 1  # 统计每个类别的正确数
#                     total_correct_category_cases += 1
#                 else:
#                     # 将列表转换为元组作为字典的键
#                     category_wrong_details[tr].append(tuple(result))  # 记录错误的分类到具体的类别中
#
# # # 打印每个类别的分类情况
# print("\n分类结果统计：")
# for category in category_total_count:
#     correct_count = category_correct_count[category]
#     total_count = category_total_count[category]
#     accuracy = correct_count / total_count if total_count > 0 else 0
#     print(f"\n类别: {category}")
#     print(f"  总数: {total_count}")
#     print(f"  正确数: {correct_count}")
#     print(f"  错误数: {total_count - correct_count}")
#     print(f"  准确率: {accuracy * 100:.2f}%")
#
#     if category in category_wrong_details:
#         wrong_instances = category_wrong_details[category]
#         wrong_instance_counts = defaultdict(int)
#         for instance in wrong_instances:
#             wrong_instance_counts[instance] += 1
#
#         print("  错误实例分布：")
#         for wrong_instance, count in wrong_instance_counts.items():
#             print(f"    错误分类为: {wrong_instance}，数量: {count}，比例: {count / total_count * 100:.2f}%")
#
# # 计算并打印最终的总准确率
# accuracy = correct_predictions / total_valid_cases if total_valid_cases > 0 else 0
# print(f"\n总体分类准确率: {accuracy * 100:.2f}% ({correct_predictions}/{total_valid_cases})")
