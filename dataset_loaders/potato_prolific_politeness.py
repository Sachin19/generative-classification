import numpy as np

from datasets import Dataset, load_dataset

LABELSTRINGS = {
    # "True_True": [
    #     ["According to a {} years old person with a {}, this email snippet demonstrates impoliteness or rudeness: "],
    #     ["According to a {} years old person with a {}, this email snippet demonstrates politeness and respect: "],
    # ],
    # "True_False": [
    #     ["According to a person with a {}, this email snippet demonstrates impoliteness or rudeness: "],
    #     ["According to a person with a {}, this email snippet demonstrates politeness and respect: "],
    # ],
    # "False_True": [
    #     ["According a person aged {}, this email snippet demonstrates impoliteness or rudeness: "],
    #     ["According a person aged {}, this email snippet demonstrates politeness and respect: "],
    # ],
    "False_False": [
        [
            "This email snippet exhibits impoliteness or rudeness: ",
            "This email snippet displays a lack of politeness or rudeness: ",
            "This email snippet is characterized by impoliteness or rudeness: ",
            "This email snippet shows a disregard for politeness or rudeness: ",
            "This email snippet conveys impoliteness or rudeness: ",
            "This email snippet contains language that is impolite or rude: ",
            "This email snippet reflects a disrespectful or rude tone: ",
            "This email snippet demonstrates a lack of courtesy or rudeness: ",
            "This email snippet uses impolite or rude language: ",
            "This email snippet expresses impoliteness or rudeness: "
        ],
        [
            "This email snippet exemplifies politeness and respect: ",
            "This email snippet demonstrates courteous and respectful language: ",
            "This email snippet displays a high level of politeness and respect: ",
            "This email snippet reflects a polite and respectful tone: ",
            "This email snippet showcases the use of polite and respectful language: ",
            "This email snippet is characterized by its politeness and respectfulness: ",
            "This email snippet conveys a sense of politeness and respect: ",
            "This email snippet is an example of proper etiquette and respect: ",
            "This email snippet demonstrates the sender's politeness and respectfulness: ",
            "This email snippet shows a considerate and respectful approach: "
        ],
    ],
    "False_True": [
        [
            "According to a {} years old person, this email snippet exhibits impoliteness or rudeness: ",
            "According to a {} years old person, this email snippet displays a lack of politeness or rudeness: ",
            "According to a {} years old person, this email snippet is characterized by impoliteness or rudeness: ",
            "According to a {} years old person, this email snippet shows a disregard for politeness or rudeness: ",
            "According to a {} years old person, this email snippet conveys impoliteness or rudeness: ",
            "According to a {} years old person, this email snippet contains language that is impolite or rude: ",
            "According to a {} years old person, this email snippet reflects a disrespectful or rude tone: ",
            "According to a {} years old person, this email snippet demonstrates a lack of courtesy or rudeness: ",
            "According to a {} years old person, this email snippet uses impolite or rude language: ",
            "According to a {} years old person, this email snippet expresses impoliteness or rudeness: "
        ],
        [
            "According to a {} years old person, this email snippet exemplifies politeness and respect: ",
            "According to a {} years old person, this email snippet demonstrates courteous and respectful language: ",
            "According to a {} years old person, this email snippet displays a high level of politeness and respect: ",
            "According to a {} years old person, this email snippet reflects a polite and respectful tone: ",
            "According to a {} years old person, this email snippet showcases the use of polite and respectful language: ",
            "According to a {} years old person, this email snippet is characterized by its politeness and respectfulness: ",
            "According to a {} years old person, this email snippet conveys a sense of politeness and respect: ",
            "According to a {} years old person, this email snippet is an example of proper etiquette and respect: ",
            "According to a {} years old person, this email snippet demonstrates the sender's politeness and respectfulness: ",
            "According to a {} years old person, this email snippet shows a considerate and respectful approach: "
        ],
    ],
    "True_False": [
        [
            "According to a person with a {}, this email snippet exhibits impoliteness or rudeness: ",
            "According to a person with a {}, this email snippet displays a lack of politeness or rudeness: ",
            "According to a person with a {}, this email snippet is characterized by impoliteness or rudeness: ",
            "According to a person with a {}, this email snippet shows a disregard for politeness or rudeness: ",
            "According to a person with a {}, this email snippet conveys impoliteness or rudeness: ",
            "According to a person with a {}, this email snippet contains language that is impolite or rude: ",
            "According to a person with a {}, this email snippet reflects a disrespectful or rude tone: ",
            "According to a person with a {}, this email snippet demonstrates a lack of courtesy or rudeness: ",
            "According to a person with a {}, this email snippet uses impolite or rude language: ",
            "According to a person with a {}, this email snippet expresses impoliteness or rudeness: "
        ],
        [
            "According to a person with a {}, this email snippet exemplifies politeness and respect: ",
            "According to a person with a {}, this email snippet demonstrates courteous and respectful language: ",
            "According to a person with a {}, this email snippet displays a high level of politeness and respect: ",
            "According to a person with a {}, this email snippet reflects a polite and respectful tone: ",
            "According to a person with a {}, this email snippet showcases the use of polite and respectful language: ",
            "According to a person with a {}, this email snippet is characterized by its politeness and respectfulness: ",
            "According to a person with a {}, this email snippet conveys a sense of politeness and respect: ",
            "According to a person with a {}, this email snippet is an example of proper etiquette and respect: ",
            "According to a person with a {}, this email snippet demonstrates the sender's politeness and respectfulness: ",
            "According to a person with a {}, this email snippet shows a considerate and respectful approach: "
        ],
    ],
    "True_True": [
        [
            "According to a {} years old person with a {}, this email snippet exhibits impoliteness or rudeness: ",
            "According to a {} years old person with a {}, this email snippet displays a lack of politeness or rudeness: ",
            "According to a {} years old person with a {}, this email snippet is characterized by impoliteness or rudeness: ",
            "According to a {} years old person with a {}, this email snippet shows a disregard for politeness or rudeness: ",
            "According to a {} years old person with a {}, this email snippet conveys impoliteness or rudeness: ",
            "According to a {} years old person with a {}, this email snippet contains language that is impolite or rude: ",
            "According to a {} years old person with a {}, this email snippet reflects a disrespectful or rude tone: ",
            "According to a {} years old person with a {}, this email snippet demonstrates a lack of courtesy or rudeness: ",
            "According to a {} years old person with a {}, this email snippet uses impolite or rude language: ",
            "According to a {} years old person with a {}, this email snippet expresses impoliteness or rudeness: "
        ],
        [
            "According to a {} years old person with a {}, this email snippet exemplifies politeness and respect: ",
            "According to a {} years old person with a {}, this email snippet demonstrates courteous and respectful language: ",
            "According to a {} years old person with a {}, this email snippet displays a high level of politeness and respect: ",
            "According to a {} years old person with a {}, this email snippet reflects a polite and respectful tone: ",
            "According to a {} years old person with a {}, this email snippet showcases the use of polite and respectful language: ",
            "According to a {} years old person with a {}, this email snippet is characterized by its politeness and respectfulness: ",
            "According to a {} years old person with a {}, this email snippet conveys a sense of politeness and respect: ",
            "According to a {} years old person with a {}, this email snippet is an example of proper etiquette and respect: ",
            "According to a {} years old person with a {}, this email snippet demonstrates the sender's politeness and respectfulness: ",
            "According to a {} years old person with a {}, this email snippet shows a considerate and respectful approach: "
        ],
    ]
}

def get_evaluation_set(num_classes=5, extreme=False, education=False, age=False):
    dataset =load_dataset("csv", data_files="/projects/tir5/users/sachink/generative-classifiers/2023/datasets/Potato-Prolific-Dataset/dataset/politeness_rating/raw_data.csv", cache_dir="/projects/tir5/users/sachink/generative-classifiers/2023/datasets/", split="train")
    dataset = dataset.filter(lambda example: example["age"] != 'Prefer not to disclose' and example["education"] not in ["Other", "Prefer not to disclose"])
    test_lines, num_labels, num_labelstrings = get_train_test_lines(dataset, num_classes, extreme, education, age)
    test_lines = list(zip(*test_lines))
    print(len(test_lines))
    return Dataset.from_dict({'id': test_lines[0], 'labelstring': test_lines[1], 'text': test_lines[2], 'labels': test_lines[3]}), num_labels, num_labelstrings


def get_train_test_lines(dataset, num_classes, extreme, education, age):
    # only train set, manually split 20% data as test
    test_indices = np.arange(len(dataset))
    np.random.seed(42)
    print(len(dataset))
    test_indices = np.random.choice(np.arange(len(dataset)), len(dataset) - int(0.8*len(dataset)), replace=False)
    test_indices.sort()
    print(len(test_indices))
    lines, num_labels, num_labelstrings = map_hf_dataset_to_list(dataset, num_classes, extreme, education, age, test_indices)

    return lines, num_labels, num_labelstrings

def map_hf_dataset_to_list(hf_dataset, num_classes, extreme, education, age, test_indices):
    def add_datapoints(datapoint, label, textid):
        lines = []
        for labelstrings in alllabelstrings:
            educationargs = [datapoint["education"]] if education else []
            ageargs = [datapoint["age"]] if age else []
            for labelstring in labelstrings:
                lines.append((textid, labelstring.format(*(educationargs + ageargs)), '"'+datapoint["text"].strip()+'"', label))
            # lines.append((datapoint["text"].strip(), int(datapoint["politeness"])))
        return lines
    
    lines = []
    alllabelstrings = LABELSTRINGS[f'{education}_{age}']
    if num_classes == 5:
        test_idx = 0
        for textid, datapoint in enumerate(hf_dataset):
            if textid != test_indices[test_idx]:
                continue
            else:
                test_idx += 1
            lines += add_datapoints(datapoint, int(datapoint["politeness"]))

    elif num_classes == 3:
        lowerlimit = 2
        upperlimit = 4
        if extreme: 
            lowerlimit = 1
            upperlimit = 5
        test_idx = 0
        for textid, datapoint in enumerate(hf_dataset):
            if textid != test_indices[test_idx]:
                continue
            else:
                test_idx += 1

            label = int(datapoint["politeness"])
            if label <= lowerlimit:
                lines += add_datapoints(datapoint, 0)
            elif label == 3:
                lines += add_datapoints(datapoint, 1)
            elif label >= upperlimit: 
                lines += add_datapoints(datapoint, 2)
    elif num_classes == 2:
        lowerlimit = 2
        upperlimit = 4
        if extreme: 
            lowerlimit = 1
            upperlimit = 5
        
        test_idx = 0
        for textid, datapoint in enumerate(hf_dataset): 
            # print(textid, test_indices[test_idx])
            if textid != test_indices[test_idx]:
                continue
            else:
                test_idx += 1
            # input(test_indices[test_idx])
            label = int(datapoint["politeness"])
            if label <= lowerlimit:
                lines += add_datapoints(datapoint, 0, textid)
            elif label >= upperlimit:
                lines += add_datapoints(datapoint, 1, textid)
    
        
    return lines, len(alllabelstrings), len(alllabelstrings[0])