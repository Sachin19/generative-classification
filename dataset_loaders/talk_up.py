from datasets import Dataset, load_dataset

LABELSTRINGS = {
    "False_False": [
        [
            "This reddit comment undermines one's confidence: ",
            "This reddit comment diminishes one's sense of empowerment: ",
            "This reddit comment has a disempowering effect: ",
            "This reddit comment makes one feel less empowered: ",
            "This reddit comment is discouraging and disempowering: ",
            "This reddit comment is disempowering: ",
            "This reddit comment leaves one feeling disempowered: ",
            "This reddit comment has a disheartening impact: ",
            "This reddit comment strips away one's sense of empowerment: ",
            "This reddit comment is demoralizing and disempowering: "
        ],
        [
            "This reddit comment empowers and uplifts: ",
            "This reddit comment instills a sense of confidence and motivation: ",
            "This reddit comment inspires and motivates: ",
            "This reddit comment fosters empowerment and positivity: ",
            "This reddit comment encourages and empowers: ",
            "This reddit comment promotes a sense of strength and empowerment: ",
            "This reddit comment uplifts and inspires: ",
            "This reddit comment empowers and motivates: ",
            "This reddit comment instills a feeling of empowerment and confidence: ",
            "This reddit comment fosters a positive and empowering atmosphere: "
        ]
    ],
    "False_True": [
        [
            "This reddit comment undermines the addressed {}'s confidence: ",
            "This reddit comment diminishes the addressed {}'s sense of empowerment: ",
            "This reddit comment has a disempowering effect on the addressed {}: ",
            "This reddit comment makes the addressed {} feel less empowered: ",
            "This reddit comment is discouraging and disempowering to the addressed {}: ",
            "This reddit comment is disempowering to the addressed {}: ",
            "This reddit comment leaves the addressed {} feeling disempowered: ",
            "This reddit comment has a disheartening impact on the addressed {}: ",
            "This reddit comment strips away the addressed {}'s sense of empowerment: ",
            "This reddit comment is demoralizing and disempowering to the addressed {}: "
        ],
        [
            "This reddit comment empowers and uplifts the addressed {}: ",
            "This reddit comment instills a sense of confidence and motivation in the addressed {}: ",
            "This reddit comment inspires and motivates the addressed {}: ",
            "This reddit comment fosters empowerment and positivity in the addressed {}: ",
            "This reddit comment encourages and empowers the addressed {}: ",
            "This reddit comment promotes a sense of strength and empowerment in the addressed {}: ",
            "This reddit comment uplifts and inspires the addressed {}: ",
            "This reddit comment empowers and motivates the addressed {}: ",
            "This reddit comment instills a feeling of empowerment and confidence in the addressed {}: ",
            "This reddit comment fosters a positive and empowering atmosphere for the addressed {}: "
        ]
    ],
    "True_False": [
        [
            "This reddit comment written by a {} undermines one's confidence: ",
            "This reddit comment written by a {} diminishes one's sense of empowerment: ",
            "This reddit comment written by a {} has a disempowering effect: ",
            "This reddit comment written by a {} makes one feel less empowered: ",
            "This reddit comment written by a {} is discouraging and disempowering: ",
            "This reddit comment written by a {} is disempowering: ",
            "This reddit comment written by a {} leaves one feeling disempowered: ",
            "This reddit comment written by a {} has a disheartening impact: ",
            "This reddit comment written by a {} strips away one's sense of empowerment: ",
            "This reddit comment written by a {} is demoralizing and disempowering: "
        ],
        [
            "This reddit comment written by a {} empowers and uplifts: ",
            "This reddit comment written by a {} instills a sense of confidence and motivation: ",
            "This reddit comment written by a {} inspires and motivates: ",
            "This reddit comment written by a {} fosters empowerment and positivity: ",
            "This reddit comment written by a {} encourages and empowers: ",
            "This reddit comment written by a {} promotes a sense of strength and empowerment: ",
            "This reddit comment written by a {} uplifts and inspires: ",
            "This reddit comment written by a {} empowers and motivates: ",
            "This reddit comment written by a {} instills a feeling of empowerment and confidence: ",
            "This reddit comment written by a {} fosters a positive and empowering atmosphere: "
        ]
    ],
    "True_True": [
        [
            "This reddit comment written by a {} undermines the addressed {}'s confidence: ",
            "This reddit comment written by a {} diminishes the addressed {}'s sense of empowerment: ",
            "This reddit comment written by a {} has a disempowering effect on the addressed {}: ",
            "This reddit comment written by a {} makes the addressed {} feel less empowered: ",
            "This reddit comment written by a {} is discouraging and disempowering to the addressed {}: ",
            "This reddit comment written by a {} is disempowering to the addressed {}: ",
            "This reddit comment written by a {} leaves the addressed {} feeling disempowered: ",
            "This reddit comment written by a {} has a disheartening impact on the addressed {}: ",
            "This reddit comment written by a {} strips away the addressed {}'s sense of empowerment: ",
            "This reddit comment written by a {} is demoralizing and disempowering to the addressed {}: "
        ],
        [
            "This reddit comment written by a {} empowers and uplifts the addressed {}: ",
            "This reddit comment written by a {} instills a sense of confidence and motivation in the addressed {}: ",
            "This reddit comment written by a {} inspires and motivates the addressed {}: ",
            "This reddit comment written by a {} fosters empowerment and positivity in the addressed {}: ",
            "This reddit comment written by a {} encourages and empowers the addressed {}: ",
            "This reddit comment written by a {} promotes a sense of strength and empowerment in the addressed {}: ",
            "This reddit comment written by a {} uplifts and inspires the addressed {}: ",
            "This reddit comment written by a {} empowers and motivates the addressed {}: ",
            "This reddit comment written by a {} instills a feeling of empowerment and confidence in the addressed {}: ",
            "This reddit comment written by a {} fosters a positive and empowering atmosphere for the addressed {}: "
        ]
    ]
}

DIRECTPROMPTS = {
    "False_False": [
        ["Predict whether this comment is disempowering: "],
        ["This reddit comment is empowering: "]
    ],
    "False_True": [
        ["This reddit comment is disempowering to the addressed {}: "],
        ["This reddit comment is empowering to the addressed {}: "]
    ],
    "True_False": [
        ["This reddit comment written by a {} is disempowering: "],
        ["This reddit comment written by a {} is empowering: "]
    ],
    "True_True": [
        ["This reddit comment written by a {} is disempowering to the addressed {}: "],
        ["This reddit comment written by a {} is empowering to the addressed {}: "]
    ],
}

def get_evaluation_set(filename, author=True, addressee=True):
    # print(type(author), type(addressee))
    # input()
    # print(author, addressee)
    author = bool(author)
    author = bool(addressee)
    dataset = load_dataset("csv", data_files=f"/projects/tir5/users/sachink/generative-classifiers/2023/datasets/{filename}", split="train", cache_dir="/projects/tir5/users/sachink/generative-classifiers/2023/datasets/").filter(lambda example: example["ambiguity"] == "no" and (not addressee or (addressee and example["responder"] is not None)))
    test_lines, num_labels, num_labelstrings = get_test_lines(dataset, author, addressee)
    test_lines = list(zip(*test_lines))
    # print(test_lines[1])
    return Dataset.from_dict({'id': test_lines[0], 'labelstring': test_lines[1], 'text': test_lines[2], 'labels': test_lines[3]}), num_labels, num_labelstrings


def get_test_lines(dataset, author, addressee):
    lines, num_labels, num_labelstrings = map_hf_dataset_to_list(dataset, author, addressee)

    return lines, num_labels, num_labelstrings

def map_hf_dataset_to_list(hf_dataset, author, addressee):
    lines = []
    alllabelstrings = LABELSTRINGS[f'{author}_{addressee}']
    textid = 0
    for textid, datapoint in enumerate(hf_dataset):
        # line[0]: input; line[1]: output
        if datapoint["empower"] == "empowering":
            label = 1
        elif datapoint["empower"] == "disempowering":
            label = 0
        else:
            continue # only binary
            
        for labelstrings in alllabelstrings:
            authorargs = [datapoint["poster"]] if author else []
            addresseeargs = [datapoint["responder"]] if addressee else []
            for labelstring in labelstrings:
                lines.append((textid, labelstring.format(*(authorargs + addresseeargs)), '"'+datapoint["post"].strip()+'"', label))

    return lines, len(alllabelstrings), len(alllabelstrings[0])
