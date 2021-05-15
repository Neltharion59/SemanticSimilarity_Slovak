# Library-like script providing pool of wrappers of all defined datasets
# Dataset is defined by its name and sub-datasets

from dataset_modification_scripts.dataset_wrapper import Dataset

dataset_pool = {
    'raw': [
        Dataset(
            "semeval-2012",
            [
                "dataset_sts_2012_MSRpar_sk.txt",
                "dataset_sts_2012_OnWN_sk.txt",
                "dataset_sts_2012_SMTnews_sk.txt",
                "dataset_sts_2012_SMTeuroparl_sk.txt"
            ]
        ),
        Dataset(
            "semeval-2013",
            [
                "dataset_sts_2013_FNWN_sk.txt",
                "dataset_sts_2013_headlines_sk.txt",
                "dataset_sts_2013_OnWN_sk.txt"
            ]
        ),
        Dataset(
            "semeval-2014",
            [
                "dataset_sts_2014_deft-forum_sk.txt",
                "dataset_sts_2014_deft-news_sk.txt",
                "dataset_sts_2014_headlines_sk.txt",
                "dataset_sts_2014_images_sk.txt",
                "dataset_sts_2014_OnWN_sk.txt",
                "dataset_sts_2014_tweet-news_sk.txt",
            ]
        ),
        Dataset(
            "semeval-2015",
            [
                "dataset_sts_2015_answers-forums_sk.txt",
                "dataset_sts_2015_answers-students_sk.txt",
                "dataset_sts_2015_belief_sk.txt",
                "dataset_sts_2015_headlines_sk.txt",
                "dataset_sts_2015_images_sk.txt"
            ]
        ),
        Dataset(
            "semeval-2016",
            [
                "dataset_sts_2016_answer-answer_sk.txt",
                "dataset_sts_2016_headlines_sk.txt",
                "dataset_sts_2016_plagiarism_sk.txt",
                "dataset_sts_2016_postediting_sk.txt",
                "dataset_sts_2016_question-question_sk.txt"
            ]
        ),
        Dataset(
            "semeval-all",
            [
                "dataset_sts_2012_MSRpar_sk.txt",
                "dataset_sts_2012_OnWN_sk.txt",
                "dataset_sts_2012_SMTnews_sk.txt",
                "dataset_sts_2012_SMTeuroparl_sk.txt",

                "dataset_sts_2013_FNWN_sk.txt",
                "dataset_sts_2013_headlines_sk.txt",
                "dataset_sts_2013_OnWN_sk.txt",

                "dataset_sts_2014_deft-forum_sk.txt",
                "dataset_sts_2014_deft-news_sk.txt",
                "dataset_sts_2014_headlines_sk.txt",
                "dataset_sts_2014_images_sk.txt",
                "dataset_sts_2014_OnWN_sk.txt",
                "dataset_sts_2014_tweet-news_sk.txt",

                "dataset_sts_2015_answers-forums_sk.txt",
                "dataset_sts_2015_answers-students_sk.txt",
                "dataset_sts_2015_belief_sk.txt",
                "dataset_sts_2015_headlines_sk.txt",
                "dataset_sts_2015_images_sk.txt",

                "dataset_sts_2016_answer-answer_sk.txt",
                "dataset_sts_2016_headlines_sk.txt",
                "dataset_sts_2016_plagiarism_sk.txt",
                "dataset_sts_2016_postediting_sk.txt",
                "dataset_sts_2016_question-question_sk.txt"
            ]
        ),
        Dataset(
            "sick",
            [
                "dataset_sick_all_sk.txt"
            ]
        ),
        Dataset(
            "all",
            [
                "dataset_sts_2012_MSRpar_sk.txt",
                "dataset_sts_2012_OnWN_sk.txt",
                "dataset_sts_2012_SMTnews_sk.txt",
                "dataset_sts_2012_SMTeuroparl_sk.txt",

                "dataset_sts_2013_FNWN_sk.txt",
                "dataset_sts_2013_headlines_sk.txt",
                "dataset_sts_2013_OnWN_sk.txt",

                "dataset_sts_2014_deft-forum_sk.txt",
                "dataset_sts_2014_deft-news_sk.txt",
                "dataset_sts_2014_headlines_sk.txt",
                "dataset_sts_2014_images_sk.txt",
                "dataset_sts_2014_OnWN_sk.txt",
                "dataset_sts_2014_tweet-news_sk.txt",

                "dataset_sts_2015_answers-forums_sk.txt",
                "dataset_sts_2015_answers-students_sk.txt",
                "dataset_sts_2015_belief_sk.txt",
                "dataset_sts_2015_headlines_sk.txt",
                "dataset_sts_2015_images_sk.txt",

                "dataset_sts_2016_answer-answer_sk.txt",
                "dataset_sts_2016_headlines_sk.txt",
                "dataset_sts_2016_plagiarism_sk.txt",
                "dataset_sts_2016_postediting_sk.txt",
                "dataset_sts_2016_question-question_sk.txt",

                "dataset_sick_all_sk.txt"
            ]
        )
    ]
}
# Create equivalent lematized entries
dataset_pool['lemma'] = [
    Dataset
    (
        dataset.name + "_lemma",
        [dataset_name.replace('_sk.txt', '_sk_lemma.txt') for dataset_name in dataset.dataset_names]
    )
    for dataset in dataset_pool['raw']
]


# Handy function to find dataset in pool with given name.
# Params: str, str
# Return: Dataset | None
def find_dataset_by_name(key, dataset_name):
    for dataset in dataset_pool[key]:
        if dataset.name == dataset_name:
            return dataset

    return None
