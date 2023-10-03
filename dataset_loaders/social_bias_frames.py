from datasets import Dataset, load_dataset
import numpy as np

LABELSTRINGS = {
    "offensive": [ 
        [
            "This {} uses innocuous language: ",
            "The language used in this {} is harmless: ",
            "The wording in this {} is inoffensive: ",
            "This {} employs language that poses no harm or offense: ",
            "The language used in this {} is benign: ",
            "The tone of this {} is innocuous and non-threatening: ",
            "The language in this {} is gentle and non-offensive: ",
            "This {} avoids using language that may cause harm or offense: ",
            "The wording in this {} is free from any malicious intent: ",
            "The language used in this {} is mild and inoffensive: ",
            "This {} utilizes language that is neutral and non-confrontational: "
        ],
        [
            "This {} uses offensive language: ",
            "The language used in this {} is offensive: ",
            "The wording in this {} is harmful: ",
            "This {} employs language that may cause harm or offense: ",
            "The language used in this {} is derogatory: ",
            "The tone of this {} is inflammatory and offensive: ",
            "The language in this {} is aggressive and disrespectful: ",
            "This {} includes language that may cause harm or offense: ",
            "The wording in this {} is filled with malicious intent: ",
            "The language used in this {} is harsh and offensive: ",
            "This {} utilizes language that is provocative and confrontational: "
        ],
    ],
    "intention": [
        ["This {} does not intend to offend: "],
        ["This {} intends to offend: "]
    ],
    "lewd": [
        [
            "This {} is free from any inappropriate or offensive content: ",
            "This {} maintains a respectful and appropriate tone: ",
            "This {} is devoid of any vulgar or offensive language: ",
            "This {} is written in a clean and respectful manner: ",
            "This {} is characterized by its cleanliness and decency: ",
            "This {} upholds a standard of cleanliness and appropriateness: ",
            "This {} reflects a positive and clean conversation: ",
            "This {} contains no objectionable or offensive material: ",
            "This {} is void of any explicit or offensive remarks: ",
            "This {} is composed in a clean and appropriate style: "
        ],
        [
            "This {} is sexually explicit: ",
            "This {} contains offensive sexual content: ",
            "This {} is vulgar and sexually suggestive: ",
            "This {} includes explicit and inappropriate language: ",
            "This {} is crude and sexually explicit: ",
            "This {} features lewd and offensive remarks: ",
            "This {} contains indecent and suggestive content: ",
            "This {} is sexually suggestive and inappropriate: ",
            "This {} includes offensive and explicit sexual references: ",
            "This {} is explicit and contains lewd content: "
        ]
    ],
    "target": [
        ["This {} is innocuous: "],
        ["This {} and is offensive to {}: "]
    ],
    "speaker_target": [
        ["This {} is innocuous to {} if written by {}: "],
        ["This {} is offensive to {} if written by {}: "]
    ],
    "annotatorx": [
        [
            "According to a {}-year-old {} {} {}, this {} is harmless: ",
            "According to a {}-year-old {} {} {}, this {} is benign: ",
            "According to a {}-year-old {} {} {}, this {} is innocuity itself: ",
            "According to a {}-year-old {} {} {}, this {} is inoffensive: ",
            "According to a {}-year-old {} {} {}, this {} is safe: ",
            "According to a {}-year-old {} {} {}, this {} is unobjectionable: ",
            "According to a {}-year-old {} {} {}, this {} is neutral: ",
            "According to a {}-year-old {} {} {}, this {} is non-controversial: ",
            "According to a {}-year-old {} {} {}, this {} is innocently intended: ",
            "According to a {}-year-old {} {} {}, this {} is not problematic: "
        ],
        [
            "According to a {}-year-old {} {} {}, this {} is objectionable: ",
            "According to a {}-year-old {} {} {}, this {} is disrespectful: ",
            "According to a {}-year-old {} {} {}, this {} is inappropriate: ",
            "According to a {}-year-old {} {} {}, this {} is offensive: ",
            "According to a {}-year-old {} {} {}, this {} is derogatory: ",
            "According to a {}-year-old {} {} {}, this {} is hurtful: ",
            "According to a {}-year-old {} {} {}, this {} is insulting: ",
            "According to a {}-year-old {} {} {}, this {} is mean-spirited: ",
            "According to a {}-year-old {} {} {}, this {} is off-putting: ",
            "According to a {}-year-old {} {} {}, this {} is inconsiderate: "
        ],
    ],
    "annotator": [
        [
            "According to a {}-year-old {} {} {}, this {} uses innocuous language: ",
            "According to a {}-year-old {} {} {}, the language used in this {} is harmless: ",
            "According to a {}-year-old {} {} {}, the wording in this {} is inoffensive: ",
            "According to a {}-year-old {} {} {}, this {} employs language that poses no harm or offense: ",
            "According to a {}-year-old {} {} {}, the language used in this {} is benign: ",
            "According to a {}-year-old {} {} {}, the tone of this {} is innocuous and non-threatening: ",
            "According to a {}-year-old {} {} {}, the language in this {} is gentle and non-offensive: ",
            "According to a {}-year-old {} {} {}, this {} avoids using language that may cause harm or offense: ",
            "According to a {}-year-old {} {} {}, the wording in this {} is free from any malicious intent: ",
            "According to a {}-year-old {} {} {}, the language used in this {} is mild and inoffensive: ",
            "According to a {}-year-old {} {} {}, this {} utilizes language that is neutral and non-confrontational: "
        ],
        [
            "According to a {}-year-old {} {} {}, this {} uses offensive language: ",
            "According to a {}-year-old {} {} {}, the language used in this {} is offensive: ",
            "According to a {}-year-old {} {} {}, the wording in this {} is harmful: ",
            "According to a {}-year-old {} {} {}, this {} employs language that may cause harm or offense: ",
            "According to a {}-year-old {} {} {}, the language used in this {} is derogatory: ",
            "According to a {}-year-old {} {} {}, the tone of this {} is inflammatory and offensive: ",
            "According to a {}-year-old {} {} {}, the language in this {} is aggressive and disrespectful: ",
            "According to a {}-year-old {} {} {}, this {} includes language that may cause harm or offense: ",
            "According to a {}-year-old {} {} {}, the wording in this {} is filled with malicious intent: ",
            "According to a {}-year-old {} {} {}, the language used in this {} is harsh and offensive: ",
            "According to a {}-year-old {} {} {}, this {} utilizes language that is provocative and confrontational: "
        ]
    ]
}

poli2desc = {
    'liberal': 'liberal',
    'mod-liberal': 'moderate liberal',
    'cons': 'conservative',
    'mod-cons': 'moderate conversative',
    'libert': 'libertarian'
}

def get_evaluation_set(whichlabels="speaker_target", ablation_domain=False):
    dataset = load_dataset("social_bias_frames", split="test", cache_dir="/projects/tir5/users/sachink/generative-classifiers/2023/datasets/").filter(lambda example: example["offensiveYN"] != "")
    if whichlabels == "annotator":
        dataset = dataset.filter(lambda example: example['annotatorGender'] != "na" and example["annotatorPolitics"] != "na" and example["annotatorPolitics"] != "other" and example["annotatorRace"] != "na" and example['annotatorRace'] != "other")
    # if whichlabels == "target":
    #     alltargets = dataset.unique("target")
    test_lines, num_labels, num_labelstrings = get_test_lines(dataset, whichlabels, ablation_domain)
    test_lines = list(zip(*test_lines))
    # print(test_lines[1][:10])
    # input()
    return Dataset.from_dict({'id': test_lines[0], 'labelstring': test_lines[1], 'text': test_lines[2], 'labels': test_lines[3]}), num_labels, num_labelstrings


def get_test_lines(dataset, whichlabels, ablation_domain):
    lines = []
    alllabelstrings = LABELSTRINGS[whichlabels]

    textid = 0
    from collections import defaultdict
    hitid2offensive = defaultdict(list)
    for textid, datapoint in enumerate(dataset):
        # line[0]: input; line[1]: output
        if "intention" in whichlabels:
            if datapoint["intentYN"] == "1.0" or datapoint["offensiveYN"] == "0.5":
                label = 1
            elif datapoint["intentYN"] == "0.0":
                label = 0
            else:
                continue # only binary, try maybe later
        elif "lewd" in whichlabels:
            if datapoint["sexYN"] == "1.0" or datapoint["offensiveYN"] == "0.5":
                label = 1
            elif datapoint["sexYN"] == "0.0":
                label = 0
            else:
                continue # only binary, try maybe later
        else: #if "offensive" in whichlabels:
            if datapoint["offensiveYN"] == "1.0" or datapoint["offensiveYN"] == "0.5":
                label = 1
            elif datapoint["offensiveYN"] == "0.0":
                label = 0
            else:
                continue # only binary, try maybe later    
        
        if not ablation_domain:
            if datapoint['dataSource'].startswith("t/"):
                source = ["tweet"]
            elif datapoint['dataSource'].startswith("r/") or datapoint['dataSource'].startswith("reddit"):
                source = ["reddit comment"]
            else:
                source = ["social-media comment"]
        else:
            source = ["text"]
        
        args = source
        if whichlabels == "speaker_target":
            target = datapoint["targetMinority"]
            speaker = datapoint["speakerMinorityYN"]
            if speaker == "0.0":
                speaker = "a person from a different minority group"
            else:
                speaker = "a person from the same minority group"
            args = [target, speaker] + source
        elif whichlabels == "annotator": # annotator
            age = datapoint["annotatorAge"]
            race = datapoint["annotatorRace"]
            politics = poli2desc[datapoint["annotatorPolitics"]]
            gender = datapoint["annotatorGender"]
            args = [age, race, politics, gender] + source

        # if datapoint['HITId']:
        #     pass
        if whichlabels not in ["annotation"]:
            if whichlabels == "offensive":
                hitid2offensive[datapoint['HITId']].append(float(datapoint["intentYN"]))
            elif whichlabels == "lewd":
                hitid2offensive[datapoint['HITId']].append(float(datapoint["sexYN"]))
            else:
                hitid2offensive[datapoint['HITId']].append(float(datapoint["offensiveYN"]))

            if len(hitid2offensive[datapoint['HITId']]) < 3:
                continue
                
            label = np.mean(hitid2offensive[datapoint['HITId']])
            if label >= 0.5:
                label = 1
            else:
                label = 0

        for labelstrings in alllabelstrings:
            for labelstring in labelstrings:
                lines.append((textid, labelstring.format(*args), '"'+datapoint["post"].strip()+'"', label))

    return lines, len(alllabelstrings), len(alllabelstrings[0])

    
