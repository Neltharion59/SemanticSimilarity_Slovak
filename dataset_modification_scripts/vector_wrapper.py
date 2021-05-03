import json


class VectorSpace:
    def __init__(self, method_name, corpus_name):
        self.method_name = method_name
        self.corpus_name = corpus_name
        self.vector_object = None

    def load_vector_object(self):
        vector_file_path = "./../resources/vectors/{}/{}".format(self.method_name, self.corpus_name)
        try:
            with open(vector_file_path, 'r', encoding='utf-8') as vector_file:
                vector_file_text = vector_file.read()
            self.vector_object = json.loads(vector_file_text)
        except FileNotFoundError:
            raise ValueError("Vector file \'{}\' not found.".format(vector_file_path))

    def access_vector_object(self):
        if self.vector_object is None:
            self.load_vector_object()
        return self.vector_object

    def unload_vector_object(self):
        self.vector_object = None
