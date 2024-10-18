import openai
import faiss
import numpy as np
import os
import re
import json
import tiktoken
import random
import pandas as pd
from tqdm import tqdm
import networkx as nx
from huggingface_hub import InferenceClient
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from COT import craniofacial_pain_syndromes, cervical_spine_pain_syndromes, thoracoabdominal_pain_syndromes, limb_and_joint_pain_syndromes, back_pain_syndromes, lumbar_degenerative_and_stenosis_syndromes, radicular_and_sciatic_pain_syndromes, generalized_pain_syndromes, neuropathic_pain_syndromes
from KG_Retrieve import main_get_category_and_level3
from gradio_client import Client
from authentication import api_key,hf_token

client = openai.OpenAI(api_key=api_key)

def get_embeddings(texts):
    embeddings = []
    for text in tqdm(texts):
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)


def get_query_embedding(query):
    return get_embeddings([query])[0]


# FAISS
def Faiss(document_embeddings, query_embedding, k):
    # index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index = faiss.IndexFlatIP(document_embeddings.shape[1])
    # index = faiss.IndexHNSWFlat(document_embeddings.shape[1])
    index.add(document_embeddings)
    _, indices = index.search(np.array([query_embedding]), k)
    print("index: ", indices)
    return indices

def extract_diagnosis(generated_text):
    diagnoses = re.findall(r'\*\*Diagnosis\*\*:\s(.*?)\n', generated_text)
    return diagnoses

def remove_parentheses(text):
    return re.sub(r'\(.*?\)', '', text).strip()

def KG_preprocess(file_path):
    kg_data = pd.read_excel(file_path, usecols=['subject', 'relation', 'object'])
    kg_data['subject'] = kg_data['subject'].apply(remove_parentheses)
    kg_data['object'] = kg_data['object'].apply(remove_parentheses)

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
    return knowledge_graph


def extract_features_from_json(file_path):
    with open(file_path, 'r') as file:
        patient_case = json.load(file)

    pain_location = patient_case.get("Pain Presentation and Description Areas of pain as per physiotherapy input", "")
    pain_symptoms = patient_case.get(
        "Pain descriptions and assorted symptoms (self-report) Associated symptoms include: parasthesia, numbness, weakness, tingling, pins and needles",
        "")

    return pain_location, pain_symptoms

level_3_to_level_2 = {
    # Respiratory System
    "acute_copd_exacerbation_infection": "respiratory_system",
    "bronchiectasis": "respiratory_system",
    "bronchiolitis": "respiratory_system",
    "bronchitis": "respiratory_system",
    "bronchospasm_acute_asthma_exacerbation": "respiratory_system",
    "pulmonary_embolism": "respiratory_system",
    "pulmonary_neoplasm": "respiratory_system",
    "spontaneous_pneumothorax": "respiratory_system",
    "urti": "respiratory_system",
    "viral_pharyngitis": "respiratory_system",
    "whooping_cough": "respiratory_system",
    "acute_laryngitis": "respiratory_system",
    "acute_pulmonary_edema": "respiratory_system",
    "croup": "respiratory_system",
    "larygospasm": "respiratory_system",
    "epiglottitis": "respiratory_system",
    "pneumonia": "respiratory_system",

    # Cardiovascular System
    "atrial_fibrillation": "cardiovascular_system",
    "myocarditis": "cardiovascular_system",
    "pericarditis": "cardiovascular_system",
    "psvt": "cardiovascular_system",
    "possible_nstemi_stemi": "cardiovascular_system",
    "stable_angina": "cardiovascular_system",
    "unstable_angina": "cardiovascular_system",
    "pulmonary_embolism": "cardiovascular_system",

    # Gastrointestinal System
    "gerd": "gastrointestinal_system",
    "boerhaave_syndrome": "gastrointestinal_system",
    "pancreatic_neoplasm": "gastrointestinal_system",
    "scombroid_food_poisoning": "gastrointestinal_system",
    "inguinal_hernia": "gastrointestinal_system",

    # Infectious Diseases
    "tuberculosis": "infectious_diseases",
    "hiv_initial_infection": "infectious_diseases",
    "ebola": "infectious_diseases",
    "influenza": "infectious_diseases",
    "chagas": "infectious_diseases",
    "acute_otitis_media": "infectious_diseases",
    "acute_rhinosinusitis": "infectious_diseases",
    "allergic_sinusitis": "infectious_diseases",
    "chronic_rhinosinusitis": "infectious_diseases",
    "pneumonia": "infectious_diseases",

    # Neurological and Muscular System
    "myasthenia_gravis": "neurological_and_muscular_system",
    "guillain_barre_syndrome": "neurological_and_muscular_system",
    "cluster_headache": "neurological_and_muscular_system",
    "acute_dystonic_reactions": "neurological_and_muscular_system",

    # Autoimmune and Immunological Diseases
    "sle": "autoimmune_and_immunological_diseases",
    "sarcoidosis": "autoimmune_and_immunological_diseases",
    "anaphylaxis": "autoimmune_and_immunological_diseases",
    "allergic_sinusitis": "autoimmune_and_immunological_diseases",

    # Psychiatric and Stress-Related Disorders
    "panic_attack": "psychiatric_and_stress_related_disorders",

    # Trauma and Injury-Related Conditions
    "spontaneous_rib_fracture": "trauma_and_injury_related_conditions",
    "spontaneous_pneumothorax": "trauma_and_injury_related_conditions",
    "inguinal_hernia": "trauma_and_injury_related_conditions",

    # Hematological Disorders
    "anemia": "hematological_disorders"
}


def get_additional_info_from_level_2(participant_no,  kg_path,top_n,match_n):
    level_2_values=main_get_category_and_level3(match_n,participant_no,top_n)
    additional_info = []
    if not level_2_values:
        print(f"No data found for Participant No.: {participant_no}")
        return None
    for level_2_value in level_2_values:
        relevant_level_3_descriptions = [desc for desc, level2 in level_3_to_level_2.items() if level2 == level_2_value]
        print("Relevant Level 3 Descriptions:", relevant_level_3_descriptions)
        if not relevant_level_3_descriptions:
            print("No Level 3 descriptions found for Level 2:", level_2_value)
            continue

        kg_data = pd.read_excel(kg_path, usecols=['subject', 'relation', 'object'])
        if kg_data.empty:
            print("Knowledge graph data is empty.")
            return None

        merged_info = {}

        for level_3 in relevant_level_3_descriptions:
            related_info = kg_data[kg_data['subject'] == level_3]

            if related_info.empty:
                print(f"No related information found in KG for: {level_3}")
            else:
                for _, row in related_info.iterrows():
                    subject = row['subject']
                    relation = row['relation'].replace('_', ' ')
                    obj = row['object']

                    if (subject, relation) in merged_info:
                        merged_info[(subject, relation)].append(obj)
                    else:
                        merged_info[(subject, relation)] = [obj]

        # K
        additional_info = []
        for (subject, relation), objects in merged_info.items():
            sentence = f"{subject} {relation} {', '.join(objects)}"
            additional_info.append(sentence)

    if not additional_info:
        print("No additional information found.")
        return None

    final_info = ', '.join(additional_info)
    print("Additional Info:", final_info)
    return final_info


def get_system_prompt_for_RAGKG():
    return '''
        You are a knowledgeable medical assistant with expertise in pain management.
        Your tasks are:
        1. Analyse and refer to the retrieved similar patients' cases and knowledge graph which may be relevant to the diagnosis and assist with new patient cases.
2. Output of "Diagnoses" must come from : acute copd exacerbation infection, bronchiectasis, bronchiolitis, bronchitis, bronchospasm acute asthma exacerbation, pulmonary embolism, pulmonary neoplasm, spontaneous pneumothorax, urti, viral pharyngitis, whooping cough, acute laryngitis, acute pulmonary edema, croup, larygospasm, epiglottitis, pneumonia, atrial fibrillation, myocarditis, pericarditis, psvt, possible nstemi stemi, stable angina, unstable angina, gerd, boerhaave syndrome, pancreatic neoplasm, scombroid food poisoning, inguinal hernia, tuberculosis, hiv initial infection, ebola, influenza, chagas, acute otitis media, acute rhinosinusitis, allergic sinusitis, chronic rhinosinusitis, myasthenia gravis, guillain barre syndrome, cluster headache, acute dystonic reactions, sle, sarcoidosis, anaphylaxis, panic attack, spontaneous rib fracture, anemia.        3. You are given differences of diagnoses of similar symptoms or pain locations. Read that information as a reference to your diagnostic if applicable.
        4. Do mind the nuance between these factors of similar diagnosis with knowledge graph information and consider it when diagnose new patient's informtation.
        5. Ensure that the recommendations are evidence-based and consider the most recent and effective practices in pain management.
        6. The output should include four specific treatment-related fields:
           - "Diagnoses (related to pain)"
           - Explanations of diagnose
           - "Pain/General Physiotherapist Treatments\nSession No.: General Overview\n- Specific interventions/treatments"
           - "Pain Psychologist Treatments"
           - "Pain Medicine Treatments"
        7. In "Diagnoses", only output the diagnosis itself. Place all other explanations and analyses (if any) into "Explanations of diagnose".
        8. You can leave Psychologist Treatments blank if not applicable for the case, leaving text "Not applicable"
        9.If you think information is needed, guide the doctor to ask further questions which following areas to distinguish between the most likely diseases: Pain restriction; Location; Symptom. Seperate answers with ",". The output should only include aspects.
        10. The output should follow this structured format:
        

    ### Diagnoses
    1. **Diagnosis**: Answer.
    2. **Explanations of diagnose**: Answer.
    
    ### Instructive question
    1. **Questions**: Answer.
    
    ### Pain/General Physiotherapist Treatments
    1. **Session No.: General Overview**
        - **Specific interventions/treatments**:
        - **Goals**:
        - **Exercises**:
        - **Manual Therapy**:
        - **Techniques**:

    2. **Exercise Recommendations from the Exercise List**:

    ### Pain Psychologist Treatments(if applicable)
    1. **Treatment 1**: 
    
    ### Pain Medicine Treatments


    ### Recommendations for Further Evaluations
    1. **Evaluation 1**:
    '''


def generate_diagnosis_report(path, query, retrieved_documents, i,top_n,match_n,model):
    system_prompt_RAGKG = get_system_prompt_for_RAGKG()
    system_prompt=system_prompt_RAGKG
    additional_info= get_additional_info_from_level_2(i ,path,top_n=top_n,match_n=match_n)

    prompt = f"{query}\nRetrieved Documents: {retrieved_documents}\nInformation from knowledge graph about relevant diagnoses, if you think the patient's disease is relevant from the suggestions provided by the atlas please refer to thoses details to distinguish similar diagnoses : {additional_info} .Now complete the tasks in that format"


    ############################################################################################openai
    if model =='gpt-4o' or 'gpt-4o-mini' or 'gpt-3.5-turbo-0125':
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    else:
        prompt=f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]"""
        LLMclient = InferenceClient(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            # "meta-llama/Llama-2-13b-chat-hf",
            # "meta-llama/Meta-Llama-3.1-70B-Instruct",
            # "meta-llama/Llama-2-13b-hf",
            # "Qwen/Qwen2-7B-Instruct",
            # "Qwen/Qwen2.5-0.5B-Instruct",
            # "mistralai/Mistral-7B-Instruct-v0.2",
            # 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            token=hf_token
        )
        response = LLMclient.text_generation(prompt=prompt,max_new_tokens=400)
        return response

def save_results_to_csv(results, output_file):
    df = pd.DataFrame(results,
                      columns=['Participant No.', 'Generated Diagnosis', 'True Diagnosis', 'Original Diagnosis'])
    df.to_csv(output_file, index=False)


folder_path=".dataset/df/train"
documents = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, file_name))]

document_embeddings_file_path='./dataset/document_embeddings.npy'

def save_embeddings(embeddings, file_path):
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    return np.load(file_path)
if os.path.exists(document_embeddings_file_path):
    document_embeddings = load_embeddings(document_embeddings_file_path)
else:
    document_embeddings = get_embeddings(documents)
    save_embeddings(document_embeddings, document_embeddings_file_path)
