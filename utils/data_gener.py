import random
import pickle as pkl

def clean_cv(cv):
    """
    Cleans the CV text by removing sentences containing specific keywords.

    Args:
        cv (str): Original CV text.

    Returns:
        str: Cleaned CV text.
    """
    sentences = cv.split('。')
    cleaned_cv = [sent for sent in sentences if not any(kw in sent for kw in ['主犯', '从犯', '主要', '次要'])]
    return '。'.join(cleaned_cv)

def generate_pairs_MASK(data_path, mask_token='MASK'):
    """
    Generates paired samples by randomly selecting principal and accomplice defendants 
    and replacing their names with a mask token.

    Args:
        data_path (str): Path to the dataset file.
        mask_token (str): Token used to replace principal or accomplice names, default is 'MASK'.

    Returns:
        tuple: Four lists - FD texts with masked names, cleaned CV texts with masked names, 
               labels, and prison terms.
    """
    data = pkl.load(open(data_path, 'rb'))
    fds, cvs, glabels, prison = [], [], [], []

    for case_id, case_data in data.items():
        defendants = list(case_data['judgment'].keys())
        prici, accomp = [], []

        for defendant in defendants:
            guilt = case_data['judgment'][defendant]['guilt']
            if guilt == 'principal':
                prici.append(defendant)
            elif guilt == 'accomplice':
                accomp.append(defendant)

        min_num = min(len(prici), len(accomp))
        prici = random.sample(prici, min_num)
        accomp = random.sample(accomp, min_num)

        for pi, ai in zip(prici, accomp):
            fds.append(case_data['FD'].replace(pi, mask_token))
            fds.append(case_data['FD'].replace(ai, mask_token))
            cvs.append(clean_cv(case_data['CV']).replace(pi, mask_token))
            cvs.append(clean_cv(case_data['CV']).replace(ai, mask_token))
            glabels.append(1)
            glabels.append(0)
            prison.append(case_data['judgment'][pi]['prison'])
            prison.append(case_data['judgment'][ai]['prison'])

    return fds, cvs, glabels, prison
