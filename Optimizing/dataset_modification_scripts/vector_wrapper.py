# Library-like script providing wrapper class for vectors

import json


# Wrapper class for constructed vectors
class VectorSpace:
    # Constructor
    # Params: str, str
    # Return: VectorSpace
    def __init__(self, method_name, corpus_name):
        self.method_name = method_name
        self.corpus_name = corpus_name
        self.vector_object = None

    # Simple method to load persisted vector object from the disk.
    # Params:
    # Return: None
    def load_vector_object(self):
        vector_file_path = "./../resources/vectors/{}/{}".format(self.method_name, self.corpus_name)
        try:
            with open(vector_file_path, 'r', encoding='utf-8') as vector_file:
                vector_file_text = vector_file.read()
            self.vector_object = json.loads(vector_file_text)
        except FileNotFoundError:
            raise ValueError("Vector file \'{}\' not found.".format(vector_file_path))

    # Access the loaded vector object from the disk (and load it if hasn't been loaded yet).
    # Params:
    # Return: dict<str, any>
    def access_vector_object(self):
        if self.vector_object is None:
            self.load_vector_object()
        return self.vector_object

    # Free the RAM by unloading the vector object.
    # Params:
    # Return: None
    def unload_vector_object(self):
        self.vector_object = None
