from . import\
    (hate_speech18, 
     ethos, tweet_eval, 
     financial_phrasebank, 
     potato_prolific_politeness, 
     hate_category,
     talk_up,
     social_bias_frames,
     glue, super_glue,
     yelp_polarity,
     yelp_review_full,
     amazon_polarity,
     amazon_review_full, 
     sst5,
     dbpedia,
     ag_news)

import numpy as np

TOKEN = "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"
TASK2LOADER = {
    "agnews": (ag_news.get_evaluation_set, ["/projects/tir5/users/sachink/generative-classifiers/label-bias/data/agnews/test.csv"]),
    "hate_speech18": (hate_speech18.get_evaluation_set, []),
    "ethos-national_origin": (ethos.get_evaluation_set, ["national_origin"]),
    "ethos-race": (ethos.get_evaluation_set, ["race"]),
    "ethos-religion": (ethos.get_evaluation_set, ["religion"]),
    "ethos-sexual_orientation": (ethos.get_evaluation_set, ["sexual_orientation"]),
    "tweet_eval-hate": (tweet_eval.get_evaluation_set, ["hate"]),
    "tweet_eval-sentiment": (tweet_eval.get_evaluation_set, ["sentiment"]),
    "tweet_eval-stance_atheism": (tweet_eval.get_evaluation_set, ["stance_atheism"]),
    "tweet_eval-stance_feminist": (tweet_eval.get_evaluation_set, ["stance_feminist"]),
    "finance_sentiment3": (financial_phrasebank.get_evaluation_set, ["sentences_allagree"]),
    "potato_prolific_politeness": (potato_prolific_politeness.get_evaluation_set, []),
    "potato_prolific_politeness_binary": (potato_prolific_politeness.get_evaluation_set, [2]),
    "potato_prolific_politeness_binary_extreme": (potato_prolific_politeness.get_evaluation_set, [2, True]),
    "hate_demographic": (hate_category.get_evaluation_set, ["demographic_category_hate_corpora.jsonl"]),
    "hate_identity": (hate_category.get_evaluation_set, ["identity_hate_corpora.jsonl"]),
    "talk_up": (talk_up.get_evaluation_set, ["talk-up.csv"]),
    "social_bias_frames": (social_bias_frames.get_evaluation_set, []),
    "sentiment2": (glue.get_evaluation_set, ["sst2"]),
    "rte": (super_glue.get_evaluation_set, ["rte"]),
    "cb": (super_glue.get_evaluation_set, ["cb"]),
    "sentiment2-yelp": (yelp_polarity.get_evaluation_set, []),
    "sentiment5-yelp": (yelp_review_full.get_evaluation_set, []),
    "sentiment2-amazon": (amazon_polarity.get_evaluation_set, []),
    "sentiment5-amazon": (amazon_review_full.get_evaluation_set, []),
    "sentiment5": (sst5.get_evaluation_set, []),
    "dbpedia": (dbpedia.get_evaluation_set, []),
    }