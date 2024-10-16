import os
import re
import json
import pandas as pd
from tqdm import tqdm
from main_MedRAG import get_query_embedding, Faiss,  extract_diagnosis, documents, document_embeddings,generate_diagnosis_report, save_results_to_csv, get_additional_info_from_level_2,KG_preprocess, get_embeddings
from authentication import ob_path,test_folder_path,ground_truth_file_path,augmented_features_path

disease_list = [
    "Head pain", "Migraine", "Trigeminal neuralgia", "Cervical spondylosis", "Chronic neck pain", "Neck pain",
    "Chest pain", "Abdominal pain", "Limb pain", "Shoulder pain", "Hip pain", "Knee pain", "Buttock pain",
    "Calf pain", "Low back pain", "Chronic low back pain", "Mechanical low back pain", "Upper back pain",
    "Degenerative disc disease", "Lumbar spondylosis", "Lumbar canal stenosis", "Spinal stenosis", "Foraminal stenosis",
    "Lumbar_radicular_pain", "Radicular pain", "Sciatica", "Lumbosacral pain", "Generalized body pain", "Fibromyalgia",
    "Musculoskeletal pain", "Myofascial pain syndrome", "Neuropathic pain", "Post-herpetic neuralgia"
]
ground_truth = pd.read_csv(ground_truth_file_path, header=0)

results = []
file_paths = os.listdir(test_folder_path)
topk=1
top_n=1
match_n=5
samplerange=range(1,552)

for i in tqdm(samplerange):

    print("topk:",topk)
    print("top_ns:",top_n)
    print("match_n:", match_n)
    print("i= ",i)
    file_path = os.path.join(test_folder_path, f"participant_{i}.json")
    if not os.path.exists(file_path):
        print(f'{i} is not found')
        continue

    with open(file_path, 'r') as file:
        new_patient_case = json.load(file)
        print(new_patient_case)

    participant_no = new_patient_case['Participant No.']
    query = json.dumps(new_patient_case)

    success = False
    while not success:
        try:
            query_embedding = get_query_embedding(query)
            indices = Faiss(document_embeddings, query_embedding,k=topk)
            retrieved_documents = [documents[i] for i in indices[0]]
            final_retrieved_info =[]
            correct_count = 0
            for retrieved_document in retrieved_documents:
                with open(retrieved_document, 'r') as file:
                    patient_case = json.load(file)
                    patient_case_json = json.dumps(patient_case)
                    patient_case_dict = json.loads(patient_case_json)
                    filtered_patient_case_dict = {
                        key: patient_case_dict[key] for key in [
                            "Processed Diagnosis",
                            "Pain Presentation and Description Areas of pain as per physiotherapy input",
                            "Pain descriptions and assorted symptoms (self-report) Associated symptoms include: parasthesia, numbness, weakness, tingling, pins and needles",
                            "Pain/General Physiotherapist Treatments (Treatments\nSession No.: General Overview\n- Specific interventions/treatments)",
                            "Pain Psychologist Treatments (Treatments)",
                            "Pain Medicine Treatments (Treatments)",
                        ] if key in patient_case_dict
                    }
                    final_retrieved_info.append(filtered_patient_case_dict)

    # ——————————————————————————————————————————————————————————————————————————————————
            true_diagnosis_row = ground_truth.loc[ground_truth['Participant No.'] == participant_no]
            if true_diagnosis_row.empty:
                print(f"True diagnosis for patient_{participant_no} not found in ground truth data")
                break

            true_diagnosis = true_diagnosis_row['Processed Diagnosis'].values[0]
            ori_truth = true_diagnosis_row['Diagnoses (related to pain)'].values[0]
            generated_report_ori = generate_diagnosis_report(augmented_features_path,query, final_retrieved_info, i,top_n=top_n,match_n=match_n)
            print(generated_report_ori)

            generated_diagnosis = re.findall(r'\*\*Diagnosis\*\*:\s*(.*?)(?:\.|\n|$)', generated_report_ori)
            if not generated_diagnosis:
                print("Generated diagnosis is either empty or not in the specified disease list. Retrying...")
                results.append([participant_no, '', true_diagnosis, ori_truth, generated_report_ori])
                break
            else:
                print("Success!!!")


            results.append([participant_no, generated_diagnosis[0], true_diagnosis, ori_truth,generated_report_ori])
            success = True
            print('________________________________________________________________')
        except Exception as e:
            print(f"Error processing patient_{participant_no}: {e}. ")

output_file = f"./test_results_topk{topk}_topn{top_n}_matchn{match_n}_{samplerange}_MedRAG.csv"

df = pd.DataFrame(results, columns=['Participant No.', 'Generated Diagnosis', 'True Diagnosis', 'Ori Truth','Generated report'])
df.to_csv(output_file, index=False)
