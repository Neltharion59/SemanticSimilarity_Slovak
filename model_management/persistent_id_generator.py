id_folder_path = "./../resources/id_generator/"


class PersistentIdGenerator:
    def __init__(self, name):
        self.name = name

    def next_id(self):
        file_path = id_folder_path + self.name + '.txt'
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                new_id = int(file.read())
            with open(file_path, 'w+', encoding='utf-8') as file:
                file.write(str(new_id + 1))
        except FileNotFoundError:
            with open(file_path, 'w+', encoding='utf-8') as file:
                file.write('2')
            new_id = 1

        return new_id

