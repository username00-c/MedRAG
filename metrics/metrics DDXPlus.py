
import pandas as pd

result_file_path = ''
ground_truth_file_path = '../dataset/test_sampled_30_per_diagnosis.csv'


def get_ACC(result_file_path, ground_truth_file_path):
    df_result = pd.read_csv(result_file_path).dropna()
    df_ground_truth = pd.read_csv(ground_truth_file_path)

    merged_df = pd.merge(df_result, df_ground_truth[['PATIENT_ID', 'Filtered_Diagnoses']],
                         left_on='Participant No.', right_on='PATIENT_ID', how='left')

    diagnosis_to_category = {
        # Organ Systems
        "acute copd exacerbation infection": (
            "organ Systems", "respiratory_system", "acute_copd_exacerbation_infection"),
        "bronchiectasis": ("Organ Systems", "respiratory_system", "bronchiectasis"),
        "bronchiolitis": ("Organ Systems", "respiratory_system", "bronchiolitis"),
        "bronchitis": ("Organ Systems", "respiratory_system", "bronchitis"),
        "bronchospasm acute asthma exacerbation": (
            "organ Systems", "respiratory_system", "bronchospasm_acute_asthma_exacerbation"),
        "pulmonary embolism": ("Organ Systems", "respiratory_system", "pulmonary_embolism"),
        "pulmonary neoplasm": ("Organ Systems", "respiratory_system", "pulmonary_neoplasm"),
        "spontaneous pneumothorax": ("Organ Systems", "respiratory_system", "spontaneous_pneumothorax"),
        "urti": ("Organ Systems", "respiratory_system", "urti"),
        "viral pharyngitis": ("Organ Systems", "respiratory_system", "viral_pharyngitis"),
        "whooping cough": ("Organ Systems", "respiratory_system", "whooping_cough"),
        "acute laryngitis": ("Organ Systems", "respiratory_system", "acute_laryngitis"),
        "acute pulmonary edema": ("Organ Systems", "respiratory_system", "acute_pulmonary_edema"),
        "croup": ("Organ Systems", "respiratory_system", "croup"),
        "larygospasm": ("Organ Systems", "respiratory_system", "larygospasm"),
        "epiglottitis": ("Organ Systems", "respiratory_system", "epiglottitis"),
        "pneumonia": ("Organ Systems", "respiratory_system", "pneumonia"),

        "atrial fibrillation": ("Organ Systems", "cardiovascular_system", "atrial_fibrillation"),
        "myocarditis": ("Organ Systems", "cardiovascular_system", "myocarditis"),
        "pericarditis": ("Organ Systems", "cardiovascular_system", "pericarditis"),
        "psvt": ("Organ Systems", "cardiovascular_system", "psvt"),
        "possible nstemi stemi": ("Organ Systems", "cardiovascular_system", "possible_nstemi_stemi"),
        "stable angina": ("Organ Systems", "cardiovascular_system", "stable_angina"),
        "unstable angina": ("Organ Systems", "cardiovascular_system", "unstable_angina"),
        "pulmonary embolism": ("Organ Systems", "cardiovascular_system", "pulmonary_embolism"),

        "gerd": ("Organ Systems", "gastrointestinal_system", "gerd"),
        "boerhaave syndrome": ("Organ Systems", "gastrointestinal_system", "boerhaave_syndrome"),
        "pancreatic neoplasm": ("Organ Systems", "gastrointestinal_system", "pancreatic_neoplasm"),
        "scombroid food poisoning": ("Organ Systems", "gastrointestinal_system", "scombroid_food_poisoning"),
        "inguinal hernia": ("Organ Systems", "gastrointestinal_system", "inguinal_hernia"),

        "myasthenia gravis": ("Organ Systems", "neurological_and_muscular_system", "myasthenia_gravis"),
        "guillain barre syndrome": ("Organ Systems", "neurological_and_muscular_system", "guillain_barre_syndrome"),
        "cluster headache": ("Organ Systems", "neurological_and_muscular_system", "cluster_headache"),
        "acute dystonic reactions": ("Organ Systems", "neurological_and_muscular_system", "acute_dystonic_reactions"),

        # Disorders
        "tuberculosis": ("Disorders", "infectious_diseases", "tuberculosis"),
        "hiv initial infection": ("Disorders", "infectious_diseases", "hiv_initial_infection"),
        "ebola": ("Disorders", "infectious_diseases", "ebola"),
        "influenza": ("Disorders", "infectious_diseases", "influenza"),
        "chagas": ("Disorders", "infectious_diseases", "chagas"),
        "acute otitis media": ("Disorders", "infectious_diseases", "acute_otitis_media"),
        "acute rhinosinusitis": ("Disorders", "infectious_diseases", "acute_rhinosinusitis"),
        "allergic sinusitis": ("Disorders", "infectious_diseases", "allergic_sinusitis"),
        "chronic rhinosinusitis": ("Disorders", "infectious_diseases", "chronic_rhinosinusitis"),
        "pneumonia": ("Disorders", "infectious_diseases", "pneumonia"),

        "sle": ("Disorders", "autoimmune_and_immunological_diseases", "sle"),
        "sarcoidosis": ("Disorders", "autoimmune_and_immunological_diseases", "sarcoidosis"),
        "anaphylaxis": ("Disorders", "autoimmune_and_immunological_diseases", "anaphylaxis"),
        "allergic sinusitis": ("Disorders", "autoimmune_and_immunological_diseases", "allergic_sinusitis"),

        "anemia": ("Disorders", "hematological_disorders", "anemia"),

        # Psychiatric and Psychological Conditions
        "panic attack": (
            "Psychiatric and Psychological Conditions", "psychiatric_and_stress_related_disorders", "panic_attack"),

        # Trauma and Injury
        "spontaneous rib fracture": (
            "Trauma and Injury", "trauma_and_injury_related_conditions", "spontaneous_rib_fracture"),
        "spontaneous pneumothorax": (
            "Trauma and Injury", "trauma_and_injury_related_conditions", "spontaneous_pneumothorax"),
        "inguinal hernia": ("Trauma and Injury", "trauma_and_injury_related_conditions", "inguinal_hernia"),
    }

    def clean_diagnosis_name(name):
        return name.replace('_', ' ').replace('.', '').replace(' /', '').replace('é', 'e').replace("-",
                                                                                                   ' ').strip().lower()

    def calculate_scores(row):
        if pd.isna(row["Filtered_Diagnoses"]):
            return {model: (0, 0, 0, row[model]) for model in ["Generated Diagnosis"]}

        true_diagnosis_raw = eval(row["Filtered_Diagnoses"])
        true_diagnosis = set(map(clean_diagnosis_name, true_diagnosis_raw))

        true_categories_level1 = set(
            clean_diagnosis_name(diagnosis_to_category.get(diag, ("unknown",))[0]) for diag in true_diagnosis)
        true_categories_level2 = set(
            clean_diagnosis_name(diagnosis_to_category.get(diag, ("unknown", "unknown"))[1]) for diag in true_diagnosis)
        true_categories_level3 = set(
            clean_diagnosis_name(diagnosis_to_category.get(diag, ("unknown", "unknown", "unknown"))[2]) for diag in
            true_diagnosis)

        results = {}
        for model in ["Generated Diagnosis"]:
            pred_diagnosis_raw = row[model].replace('"', '').split(', ')
            pred_diagnosis = set(map(clean_diagnosis_name, pred_diagnosis_raw))

            pred_categories_level1 = set(
                clean_diagnosis_name(diagnosis_to_category.get(diag, ("unknown",))[0]) for diag in pred_diagnosis)
            pred_categories_level2 = set(
                clean_diagnosis_name(diagnosis_to_category.get(diag, ("unknown", "unknown"))[1]) for diag in
                pred_diagnosis)
            pred_categories_level3 = set(
                clean_diagnosis_name(diagnosis_to_category.get(diag, ("unknown", "unknown", "unknown"))[2]) for diag in
                pred_diagnosis)

            level1_score = int(bool(true_categories_level1 & pred_categories_level1))
            level2_score = int(bool(true_categories_level2 & pred_categories_level2))
            level3_score = int(bool(true_categories_level3 & pred_categories_level3))

            results[model] = (level1_score, level2_score, level3_score)

        return results

    # 计算准确率
    scores = merged_df.apply(calculate_scores, axis=1)

    # 计算每个级别的准确度
    for model in ["Generated Diagnosis"]:
        level1_correct = sum(scores[idx][model][0] for idx in scores.index)
        level2_correct = sum(scores[idx][model][1] for idx in scores.index)
        level3_correct = sum(scores[idx][model][2] for idx in scores.index)
        total = len(merged_df)

        print(f"{model} - Level 1 accuracy: {level1_correct / total:.4f}")
        print(f"{model} - Level 2 accuracy: {level2_correct / total:.4f}")
        print(f"{model} - Level 3 accuracy: {level3_correct / total:.4f}")

    df_result.to_csv(result_file_path, index=False)


get_ACC(result_file_path, ground_truth_file_path)