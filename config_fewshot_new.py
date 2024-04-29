EXAMPLEFORMAT = "\"{text}\""
EXAMPLEFORMAT2 = "Premise: {text1}. Hypothesis {text2}"
EXAMPLEFORMAT2ENTAIL = "This premise: {text1} entails this hypothesis {text2}."
EXAMPLEFORMAT2NOTENTAIL = "This premise: {text1} contradicts this hypothesis {text2}."
EXAMPLEFORMAT_SPACE = " \"{text}\""
EXAMPLEFORMAT2_SPACE = " Premise: {text1}. Hypothesis {text2}"
EXAMPLEFORMAT_SPACE2ENTAIL = " This premise: \"{text1}\" entails this hypothesis {text2}."
EXAMPLEFORMAT_SPACE2NOTENTAIL = " This premise: \"{text1}\" does not entail  this hypothesis {text2}."


TASK2LABELSTRINGS = {
    "rte_fewshot": [
        "{premise}\nquestion: {hypothesis} true or false? answer: {label}"
    ],
    "sick_fewshot": [
        "{premise}\nquestion: {hypothesis}. entailment, neutral, or contradiction? answer: {label}"
    ],
    "cb_fewshot": [
        "{premise}\nquestion: {hypothesis}. true, false, or neither? answer: {label}"
    ],
    "wnli_fewshot": [
        "{premise}\nquestion: {hypothesis}. True or False? answer: {label}"
    ],
    "mnli_fewshot": [
        "{premise}\nquestion: {hypothesis}. entailment, neutral, or contradiction answer: {label}"
    ],
    "arc_easy_fewshot": [
        "Question:\n{question}\nChoices:\n{choices}Answer: {label}", # choices formatted as "{1}. {choice1}\n{2}. {choice2}"
        "{label}: {choice}\n"
    ],
    "arc_challenge_fewshot": [
        "Question:\n{question}\nChoices:\n{choices}Answer: {label}", # choices formatted as "{1}. {choice1}\n{2}. {choice2}"
        "{label}: {choice}\n"
    ],
    "openbookqa_fewshot": [
        "Question:\n{question}\nChoices:\n{choices}Answer: {label}", # choices formatted as "{1}. {choice1}\n{2}. {choice2}"
        "{label}: {choice}\n"
    ],
    "commonsense_qa_fewshot": [
        "Question: {question}\nChoices:\n{choices}Answer: {label}", # choices formatted as "{1}. {choice1}\n{2}. {choice2}"
        "{label}: {choice}\n"
    ],
    "copa_fewshot": [
        "Question: {question}\nChoices:\n{choices}Answer: {label}", # choices formatted as "{1}. {choice1}\n{2}. {choice2}"
        "{label}: {choice}\n"
    ],
    "piqa_fewshot": [
        "Question: {question}\nChoices:\n{choices}Answer: {label}", # choices formatted as "{1}. {choice1}\n{2}. {choice2}"
        "{label}: {choice}\n"
    ],
    "winogrande_fewshot": [
        "Question: {question}\nChoices:\n{choices}Answer: {label}", # choices formatted as "{1}. {choice1}\n{2}. {choice2}"
        "{label}: {choice}\n"
    ],
    "mmlu_fewshot": [
        "Question: {question}\nChoices:\n{choices}Answer: {label}",
        "{label}: {choice}\n"
    ],
    "boolq_fewshot": [
        "Passage:\n{passage}\nQuestion:\n{question}\nChoices:\n{choices}Answer: {label}", # choices formatted as "{1}. {choice1}\n{2}. {choice2}"
        "{label}: {choice}\n"
    ],
    "sst2_fewshot": [
        "Statement: {sentence}\nSentiment: {label}"
    ],
    "rte_hGivenP_fewshot": [
        "{premise}\ntrue or false? answer: true. question: {hypothesis}",
        "{premise}\ntrue or false? answer: false. question: {hypothesis}",
    ],
    "sick_hGivenP_fewshot": [
        "This: {premise}, entails this: {hypothesis}",
        "The premise: {premise} is neutral to the hypothesis: {hypothesis}",
        "This: {premise}, contradicts this: {hypothesis}",
    ],
}

TASK2ARGS ={
    "sst2": {},
    "sst5": {},
    "yelp2": {},
    "yelp5": {},
    "poem_sentiment": {},
    "yahoo_answers_topics": {},
    "emotion6": {},
    "agnews": {},
    "trec": {},
    "subjectivity": {},
    "boolq": {},
    "jigsaw": {},
    "hate_speech18": {},
}

# ["It is about World. "],
# ["It is about Sports. "],
# ["It is about Business. "],
# ["It is about Technology. "]

"""
[
            "This: <break>\"{text1}\"<break> contradicts this: <break>\"{text2}\"",
            "The premise: <break>\"{text1}\"<break> undermines the hypothesis: <break>\"{text2}\"",
            "From the premise: <break>\"{text1}\"<break> we cannot derive the hypothesis: <break>\"{text2}\"",
            "The premise: <break>\"{text1}\"<break> does not suggest the following hypothesis: <break>\"{text2}\"",
            "The premise: <break>\"{text1}\"<break> cannot infer the hypothesis: <break>\"{text2}\"",
            "The argument: <break>\"{text1}\"<break> doesn't result in the hypothesis: <break>\"{text2}\"",
            "The foundation: <break>\"{text1}\"<break> doesn't give rise to: <break>\"{text2}\"",
            "The premise: <break>\"{text1}\"<break> omits the hypothesis: <break>\"{text2}\"",
            "This premise: <break>\"{text1}\"<break> disagrees with this hypothesis: <break>\"{text2}\"",
            "The premise: <break>\"{text1}\"<break> opposes the hypothesis: <break>\"{text2}\"",
            "The premise <break>\"{text1}\"<break> clashes with the hypothesis <break>\"{text2}\"",
            # "Premise {text1} is in conflict with hypothesis {text2}",
            # "The basis {text1} opposes the theory {text2}",
            # "The premise {text1}, disputes the hypothesis {text2}",
            # "The starting point {text1} negates the idea {text2}",
            # "Initial concept {text1} contradicts proposed {text2}",
            # "Foundational {text1} counters hypothesis {text2}",
            # "The assertion {text1} challenges the hypothesis {text2}",
            # "Premise {text1} refutes the theory {text2}",
            # "Core idea {text1} disputes hypothesis {text2}",
            # "The stance {text1} contradicts the proposition {text2}",
            # "Foundational belief {text1} opposes {text2} as hypothesis.",
            # "The claim {text1} counters the theory {text2}",
            # "Underlying premise {text1} conflicts with {text2}",
            # "Starting belief {text1} contradicts the thought {text2}",
            # "Basic assumption {text1} challenges hypothesis {text2}",
            # "Premise {text1} disputes the speculation {text2}",
            # "The idea {text1} contests hypothesis {text2}",
            # "Original concept {text1} is at odds with {text2}",
            # "The principle {text1} refutes the hypothesis {text2}",
            # "The base {text1} contradicts the guess {text2}",
            # "Premise {text1} denies the theory {text2}",
            # "The idea {text1} rejects the hypothesis {text2}",
            # "Starting point {text1} opposes the notion {text2}",
            # "The assertion {text1} refutes the concept {text2}",
            # "Fundamental {text1} conflicts with hypothesis {text2}",
            # "Original premise {text1} challenges the idea {text2}",
            # "The claim {text1} discredits the theory {text2}",
            # "Basic idea {text1} is against hypothesis {text2}",
            # "The thought {text1} counters the assumption {text2}",
            # "Founding premise {text1} disputes the theory {text2}",
            # "Core assertion {text1} contradicts the guess {text2}",
            # "Main premise {text1} negates the hypothesis {text2}",
            # "Initial theory {text1} conflicts with idea {text2}",
            # "The notion {text1} counteracts the theory {text2}",
            # "Basic premise {text1} opposes the thought {text2}",
            # "Starting assertion {text1} clashes with {text2}",
            # "Fundamental idea {text1} refutes the theory {text2}",
            # "Key premise {text1} disagrees with hypothesis {text2}",
            # "Underlying assumption {text1} challenges {text2}"
        ],
        [
            "This: <break>\"{text1}\"<break> entails this: <break>\"{text2}\"",
            "The premise: <break>\"{text1}\"<break> supports the hypothesis: <break>\"{text2}\"",
            "From the premise: <break>\"{text1}\"<break> we can derive the hypothesis: <break>\"{text2}\"",
            "The premise: <break>\"{text1}\"<break> suggests the following hypothesis: <break>\"{text2}\"",
            "The premise: <break>\"{text1}\"<break> can infer the hypothesis: <break>\"{text2}\"",
            "The argument: <break>\"{text1}\"<break> does result in the hypothesis: <break>\"{text2}\"",
            "This: <break>\"{text1}\"<break> does give rise to: <break>\"{text2}\"",
            "The premise: <break>\"{text1}\"<break> warrants the hypothesis: <break>\"{text2}\"",
            "This premise: <break>\"{text1}\"<break> agrees with this hypothesis: <break>\"{text2}\"",
            "The premise: <break>\"{text1}\"<break> implies the hypothesis: <break>\"{text2}\"", 
            "Given this premise: <break>\"{text1}\"<break> the following hypothesis does entail: <break>\"{text2}\"",
            # "The argument: {text1} as the premise results in the hypothesis: {text2}",
            # "The proposition: {text1} leads naturally to the hypothesis: {text2}",
            # "The foundation {text1} bolsters the theory {text2}",
            # "The basic idea {text1} reinforces the supposition {text2}",
            # "The underlying principle {text1} upholds the proposition {text2}",
            # "The concept {text1} fortifies the assumption {text2}",
            # "The groundwork {text1} backs the speculation {text2}",
            # "The assumption {text1} endorses the hypothesis {text2}",
            # "The argument {text1} strengthens the thesis {text2}",
            # "The notion {text1} corroborates the conjecture {text2}",
            # "The rationale {text1} validates the presumption {text2}",
            # "The basis {text1} substantiates the inference {text2}",
            # "The proposition {text1} confirms the guess {text2}",
            # "The starting point {text1} advocates for the theory {text2}",
            # "The core concept {text1} supports the line of reasoning {text2}",
            # "The underlying assumption {text1} gives credence to the idea {text2}",
            # "The foundational belief {text1} aligns with the hypothesis {text2}",
            # "The key idea {text1} is in harmony with the hypothesis {text2}",
            # "The basic assertion {text1} leads to the hypothesis {text2}",
            # "The principle argument {text1} is congruent with the hypothesis {text2}",
            # "The initial theory {text1} lends weight to the hypothesis {text2}",
            # "The starting assertion {text1} paves the way for the hypothesis {text2}",
            # "The base idea {text1} is the precursor to the hypothesis {text2}",
            # "The key premise {text1} lays the groundwork for the hypothesis {text2}",
            # "The primary assumption {text1} sets the stage for the hypothesis {text2}",
            # "The original proposition {text1} feeds into the hypothesis {text2}",
            # "The core belief {text1} acts as a foundation for the hypothesis {text2}",
            # "This: {text1} leads to {text2}",
            # "The basic concept {text1} provides support for the hypothesis {text2}",
            # "The initial argument {text1} forms the basis of the hypothesis {text2}",
            # "The ground rule {text1} is aligned with the hypothesis {text2}",
            # "The principal idea {text1} serves as a basis for the hypothesis {text2}",
            # "The central thesis {text1} is the precursor for the hypothesis {text2}",
            # "The ground rule {text1} is aligned with the hypothesis {text2}",
            # "The premise {text1} aligns with the hypothesis {text2}"
            # "The premise {text1} points to the hypothesis {text2}",
            # "This: {text1} strengthens {text2}",
            # "This: {text1} substantiates {text2}",
            # "The main argument {text1} is a precursor to the hypothesis {text2}"
        ]
"""



"""
[
            "This: \"{text1}\" contradicts this: \"{text2}\"",
            "The premise: \"{text1}\" undermines the hypothesis: \"{text2}\"",
            "From the premise: \"{text1}\" we cannot derive the hypothesis: \"{text2}\"",
            "The premise: \"{text1}\" does not suggest the following hypothesis: \"{text2}\"",
            "The premise: \"{text1}\" cannot infer the hypothesis: \"{text2}\"",
            "The argument: \"{text1}\" doesn't result in the hypothesis: \"{text2}\"",
            "The foundation: \"{text1}\" doesn't give rise to: \"{text2}\"",
            "The premise: \"{text1}\" omits the hypothesis: \"{text2}\"",
            "This premise: \"{text1}\" disagrees with this hypothesis: \"{text2}\"",
            "The premise: \"{text1}\" opposes the hypothesis: \"{text2}\"",
            "The premise \"{text1}\" clashes with the hypothesis \"{text2}\"",
            # "Premise {text1} is in conflict with hypothesis {text2}",
            # "The basis {text1} opposes the theory {text2}",
            # "The premise {text1}, disputes the hypothesis {text2}",
            # "The starting point {text1} negates the idea {text2}",
            # "Initial concept {text1} contradicts proposed {text2}",
            # "Foundational {text1} counters hypothesis {text2}",
            # "The assertion {text1} challenges the hypothesis {text2}",
            # "Premise {text1} refutes the theory {text2}",
            # "Core idea {text1} disputes hypothesis {text2}",
            # "The stance {text1} contradicts the proposition {text2}",
            # "Foundational belief {text1} opposes {text2} as hypothesis.",
            # "The claim {text1} counters the theory {text2}",
            # "Underlying premise {text1} conflicts with {text2}",
            # "Starting belief {text1} contradicts the thought {text2}",
            # "Basic assumption {text1} challenges hypothesis {text2}",
            # "Premise {text1} disputes the speculation {text2}",
            # "The idea {text1} contests hypothesis {text2}",
            # "Original concept {text1} is at odds with {text2}",
            # "The principle {text1} refutes the hypothesis {text2}",
            # "The base {text1} contradicts the guess {text2}",
            # "Premise {text1} denies the theory {text2}",
            # "The idea {text1} rejects the hypothesis {text2}",
            # "Starting point {text1} opposes the notion {text2}",
            # "The assertion {text1} refutes the concept {text2}",
            # "Fundamental {text1} conflicts with hypothesis {text2}",
            # "Original premise {text1} challenges the idea {text2}",
            # "The claim {text1} discredits the theory {text2}",
            # "Basic idea {text1} is against hypothesis {text2}",
            # "The thought {text1} counters the assumption {text2}",
            # "Founding premise {text1} disputes the theory {text2}",
            # "Core assertion {text1} contradicts the guess {text2}",
            # "Main premise {text1} negates the hypothesis {text2}",
            # "Initial theory {text1} conflicts with idea {text2}",
            # "The notion {text1} counteracts the theory {text2}",
            # "Basic premise {text1} opposes the thought {text2}",
            # "Starting assertion {text1} clashes with {text2}",
            # "Fundamental idea {text1} refutes the theory {text2}",
            # "Key premise {text1} disagrees with hypothesis {text2}",
            # "Underlying assumption {text1} challenges {text2}"
        ],
        [
            "This: \"{text1}\" entails this: \"{text2}\"",
            "The premise: \"{text1}\" supports the hypothesis: \"{text2}\"",
            "From the premise: \"{text1}\" we can derive the hypothesis: \"{text2}\"",
            "The premise: \"{text1}\" suggests the following hypothesis: \"{text2}\"",
            "The premise: \"{text1}\" can infer the hypothesis: \"{text2}\"",
            "The argument: \"{text1}\" does result in the hypothesis: \"{text2}\"",
            "This: \"{text1}\" does give rise to: \"{text2}\"",
            "The premise: \"{text1}\" warrants the hypothesis: \"{text2}\"",
            "This premise: \"{text1}\" agrees with this hypothesis: \"{text2}\"",
            "The premise: \"{text1}\" implies the hypothesis: \"{text2}\"", 
            "Given this premise: \"{text1}\" the following hypothesis does entail: \"{text2}\"",
            # "The argument: {text1} as the premise results in the hypothesis: {text2}",
            # "The proposition: {text1} leads naturally to the hypothesis: {text2}",
            # "The foundation {text1} bolsters the theory {text2}",
            # "The basic idea {text1} reinforces the supposition {text2}",
            # "The underlying principle {text1} upholds the proposition {text2}",
            # "The concept {text1} fortifies the assumption {text2}",
            # "The groundwork {text1} backs the speculation {text2}",
            # "The assumption {text1} endorses the hypothesis {text2}",
            # "The argument {text1} strengthens the thesis {text2}",
            # "The notion {text1} corroborates the conjecture {text2}",
            # "The rationale {text1} validates the presumption {text2}",
            # "The basis {text1} substantiates the inference {text2}",
            # "The proposition {text1} confirms the guess {text2}",
            # "The starting point {text1} advocates for the theory {text2}",
            # "The core concept {text1} supports the line of reasoning {text2}",
            # "The underlying assumption {text1} gives credence to the idea {text2}",
            # "The foundational belief {text1} aligns with the hypothesis {text2}",
            # "The key idea {text1} is in harmony with the hypothesis {text2}",
            # "The basic assertion {text1} leads to the hypothesis {text2}",
            # "The principle argument {text1} is congruent with the hypothesis {text2}",
            # "The initial theory {text1} lends weight to the hypothesis {text2}",
            # "The starting assertion {text1} paves the way for the hypothesis {text2}",
            # "The base idea {text1} is the precursor to the hypothesis {text2}",
            # "The key premise {text1} lays the groundwork for the hypothesis {text2}",
            # "The primary assumption {text1} sets the stage for the hypothesis {text2}",
            # "The original proposition {text1} feeds into the hypothesis {text2}",
            # "The core belief {text1} acts as a foundation for the hypothesis {text2}",
            # "This: {text1} leads to {text2}",
            # "The basic concept {text1} provides support for the hypothesis {text2}",
            # "The initial argument {text1} forms the basis of the hypothesis {text2}",
            # "The ground rule {text1} is aligned with the hypothesis {text2}",
            # "The principal idea {text1} serves as a basis for the hypothesis {text2}",
            # "The central thesis {text1} is the precursor for the hypothesis {text2}",
            # "The ground rule {text1} is aligned with the hypothesis {text2}",
            # "The premise {text1} aligns with the hypothesis {text2}"
            # "The premise {text1} points to the hypothesis {text2}",
            # "This: {text1} strengthens {text2}",
            # "This: {text1} substantiates {text2}",
            # "The main argument {text1} is a precursor to the hypothesis {text2}"
        ]
"""

"""
[
            "This: {text1} contradicts this: {text2}",
            "The premise: {text1} undermines the hypothesis: {text2}",
            "From the premise: {text1} we cannot derive the hypothesis: {text2}",
            "The premise: {text1} does not suggest the following hypothesis: {text2}",
            "The premise: {text1} cannot infer the hypothesis: {text2}",
            "The argument: {text1} doesn't result in the hypothesis: {text2}",
            "The foundation: {text1} doesn't give rise to: {text2}",
            "The premise: {text1} omits the hypothesis: {text2}",
            "This premise: {text1} disagrees with this hypothesis: {text2}",
            "The premise: {text1} opposes the hypothesis: {text2}",
            "The premise {text1} clashes with the hypothesis {text2}",
            # "Premise {text1} is in conflict with hypothesis {text2}",
            # "The basis {text1} opposes the theory {text2}",
            # "The premise {text1}, disputes the hypothesis {text2}",
            # "The starting point {text1} negates the idea {text2}",
            # "Initial concept {text1} contradicts proposed {text2}",
            # "Foundational {text1} counters hypothesis {text2}",
            # "The assertion {text1} challenges the hypothesis {text2}",
            # "Premise {text1} refutes the theory {text2}",
            # "Core idea {text1} disputes hypothesis {text2}",
            # "The stance {text1} contradicts the proposition {text2}",
            # "Foundational belief {text1} opposes {text2} as hypothesis.",
            # "The claim {text1} counters the theory {text2}",
            # "Underlying premise {text1} conflicts with {text2}",
            # "Starting belief {text1} contradicts the thought {text2}",
            # "Basic assumption {text1} challenges hypothesis {text2}",
            # "Premise {text1} disputes the speculation {text2}",
            # "The idea {text1} contests hypothesis {text2}",
            # "Original concept {text1} is at odds with {text2}",
            # "The principle {text1} refutes the hypothesis {text2}",
            # "The base {text1} contradicts the guess {text2}",
            # "Premise {text1} denies the theory {text2}",
            # "The idea {text1} rejects the hypothesis {text2}",
            # "Starting point {text1} opposes the notion {text2}",
            # "The assertion {text1} refutes the concept {text2}",
            # "Fundamental {text1} conflicts with hypothesis {text2}",
            # "Original premise {text1} challenges the idea {text2}",
            # "The claim {text1} discredits the theory {text2}",
            # "Basic idea {text1} is against hypothesis {text2}",
            # "The thought {text1} counters the assumption {text2}",
            # "Founding premise {text1} disputes the theory {text2}",
            # "Core assertion {text1} contradicts the guess {text2}",
            # "Main premise {text1} negates the hypothesis {text2}",
            # "Initial theory {text1} conflicts with idea {text2}",
            # "The notion {text1} counteracts the theory {text2}",
            # "Basic premise {text1} opposes the thought {text2}",
            # "Starting assertion {text1} clashes with {text2}",
            # "Fundamental idea {text1} refutes the theory {text2}",
            # "Key premise {text1} disagrees with hypothesis {text2}",
            # "Underlying assumption {text1} challenges {text2}"
        ],
        [
            "This: {text1} entails this: {text2}",
            "The premise: {text1} supports the hypothesis: {text2}",
            "From the premise: {text1} we can derive the hypothesis: {text2}",
            "The premise: {text1} suggests the following hypothesis: {text2}",
            "The premise: {text1} can infer the hypothesis: {text2}",
            "The argument: {text1} does result in the hypothesis: {text2}",
            "This: {text1} does give rise to: {text2}",
            "The premise: {text1} warrants the hypothesis: {text2}",
            "This premise: {text1} agrees with this hypothesis: {text2}",
            "The premise: {text1} implies the hypothesis: {text2}", 
            "Given this premise: {text1} the following hypothesis does entail: {text2}",
            # "The argument: {text1} as the premise results in the hypothesis: {text2}",
            # "The proposition: {text1} leads naturally to the hypothesis: {text2}",
            # "The foundation {text1} bolsters the theory {text2}",
            # "The basic idea {text1} reinforces the supposition {text2}",
            # "The underlying principle {text1} upholds the proposition {text2}",
            # "The concept {text1} fortifies the assumption {text2}",
            # "The groundwork {text1} backs the speculation {text2}",
            # "The assumption {text1} endorses the hypothesis {text2}",
            # "The argument {text1} strengthens the thesis {text2}",
            # "The notion {text1} corroborates the conjecture {text2}",
            # "The rationale {text1} validates the presumption {text2}",
            # "The basis {text1} substantiates the inference {text2}",
            # "The proposition {text1} confirms the guess {text2}",
            # "The starting point {text1} advocates for the theory {text2}",
            # "The core concept {text1} supports the line of reasoning {text2}",
            # "The underlying assumption {text1} gives credence to the idea {text2}",
            # "The foundational belief {text1} aligns with the hypothesis {text2}",
            # "The key idea {text1} is in harmony with the hypothesis {text2}",
            # "The basic assertion {text1} leads to the hypothesis {text2}",
            # "The principle argument {text1} is congruent with the hypothesis {text2}",
            # "The initial theory {text1} lends weight to the hypothesis {text2}",
            # "The starting assertion {text1} paves the way for the hypothesis {text2}",
            # "The base idea {text1} is the precursor to the hypothesis {text2}",
            # "The key premise {text1} lays the groundwork for the hypothesis {text2}",
            # "The primary assumption {text1} sets the stage for the hypothesis {text2}",
            # "The original proposition {text1} feeds into the hypothesis {text2}",
            # "The core belief {text1} acts as a foundation for the hypothesis {text2}",
            # "This: {text1} leads to {text2}",
            # "The basic concept {text1} provides support for the hypothesis {text2}",
            # "The initial argument {text1} forms the basis of the hypothesis {text2}",
            # "The ground rule {text1} is aligned with the hypothesis {text2}",
            # "The principal idea {text1} serves as a basis for the hypothesis {text2}",
            # "The central thesis {text1} is the precursor for the hypothesis {text2}",
            # "The ground rule {text1} is aligned with the hypothesis {text2}",
            # "The premise {text1} aligns with the hypothesis {text2}"
            # "The premise {text1} points to the hypothesis {text2}",
            # "This: {text1} strengthens {text2}",
            # "This: {text1} substantiates {text2}",
            # "The main argument {text1} is a precursor to the hypothesis {text2}"
        ]
"""