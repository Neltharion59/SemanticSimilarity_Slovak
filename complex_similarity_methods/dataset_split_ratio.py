class DatasetSplitRatio:
    def __init__(self, train_part, validation_part):
        sum_of_parts = train_part + validation_part

        self.train_ratio = train_part / sum_of_parts
        self.validation_ratio = validation_part / sum_of_parts
