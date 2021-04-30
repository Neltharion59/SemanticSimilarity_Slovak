class Corpus:
    def __init__(self, name):
        self.name = name

        try:
            with open("./../resources/corpora/{}".format(self.name), 'r', encoding='utf-8') as file:
                self.size = len(file.readlines())
        except FileNotFoundError:
            self.size = 0

    def lines(self):
        with open("./../resources/corpora/{}".format(self.name), 'r', encoding='utf-8') as file:
            for line in file:
                yield line
